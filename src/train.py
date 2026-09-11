"""Train, tune, validate and persist the real-data transaction fraud benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.data_processing import inspect_dataset, make_schema, read_csv, split_dataset, validate_features
from src.dataset import load_training_frame
from src.eda import create_eda
from src.evaluate import curves, evaluate, histogram, optimize_threshold
from src.feature_engineering import preprocessing


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def candidates(schema, seed, profile="recall"):
    if profile == "recall":
        forest = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=seed)
        numeric = all(item["type"] == "number" for item in schema)
        # Do not multiply class weighting by oversampling: the sampler already changes prevalence.
        choices = [
            ("Random Forest", clone(forest).set_params(class_weight="balanced_subsample"), "passthrough",
             {"classifier__max_depth": [10, 18]}, "class weights baseline"),
            ("Random Forest · random oversampling", clone(forest), RandomOverSampler(random_state=seed),
             {"classifier__max_depth": [10, 18], "sampler__sampling_strategy": [0.1, 0.25]},
             "training-only random oversampling; minority/majority ratio selected by CV"),
        ]
        if numeric:
            choices.extend([
                ("Random Forest · SMOTE", clone(forest), SMOTE(k_neighbors=3, random_state=seed),
                 {"classifier__max_depth": [10, 18], "sampler__sampling_strategy": [0.1, 0.25]},
                 "training-only SMOTE; minority/majority ratio selected by CV"),
                ("XGBoost · SMOTE", XGBClassifier(n_estimators=220, learning_rate=.07, tree_method="hist", n_jobs=4,
                                                 eval_metric="logloss", random_state=seed, subsample=.85, colsample_bytree=.85),
                 SMOTE(k_neighbors=3, random_state=seed),
                 {"classifier__max_depth": [3, 5], "sampler__sampling_strategy": [0.1, 0.25]},
                 "training-only SMOTE; minority/majority ratio selected by CV"),
            ])
        else:
            boosting = XGBClassifier(n_estimators=220, learning_rate=.07, tree_method="hist", n_jobs=4,
                                     eval_metric="logloss", random_state=seed, subsample=.85, colsample_bytree=.85)
            choices.extend([
                ("XGBoost", clone(boosting), "passthrough", {"classifier__max_depth": [3, 5]}, "unweighted boosting baseline"),
                ("XGBoost · random oversampling", clone(boosting), RandomOverSampler(random_state=seed),
                 {"classifier__max_depth": [3, 5], "sampler__sampling_strategy": [.1, .25]},
                 "training-only random oversampling; valid categorical rows; ratio selected by CV"),
            ])
        return choices
    if profile != "benchmark":
        raise ValueError("Training profile must be 'recall' or 'benchmark'.")
    logistic = LogisticRegression(max_iter=700, solver="lbfgs", random_state=seed)
    choices = [
        ("Logistic Regression · weighted", clone(logistic).set_params(class_weight="balanced"), "passthrough", {"classifier__C": [0.1, 1.0]}, "class weights"),
        ("Logistic Regression · undersampling", clone(logistic), RandomUnderSampler(sampling_strategy=0.1, random_state=seed), {"classifier__C": [0.1, 1.0]}, "training-only random undersampling to 1:10"),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=4, random_state=seed), "passthrough", {"classifier__max_depth": [10, 18]}, "class weights"),
        ("Histogram Gradient Boosting", HistGradientBoostingClassifier(max_iter=160, class_weight="balanced", early_stopping=False, random_state=seed), "passthrough", {"classifier__max_leaf_nodes": [15, 31]}, "class weights"),
        ("XGBoost", XGBClassifier(n_estimators=220, learning_rate=0.07, tree_method="hist", n_jobs=4, eval_metric="logloss", random_state=seed, subsample=.85, colsample_bytree=.85), "passthrough", {"classifier__max_depth": [3, 5]}, "unweighted boosting baseline"),
    ]
    if all(item["type"] == "number" for item in schema):
        # Avoid interpolating one-hot categorical codes. Mixed data uses weights/undersampling.
        choices.insert(1, ("Logistic Regression · SMOTE", clone(logistic), SMOTE(sampling_strategy=0.1, k_neighbors=3, random_state=seed), {"classifier__C": [0.1, 1.0]}, "training-only SMOTE to 1:10"))
    return choices


def select_candidate(results, profile):
    """Only validation metrics select the model. Test results must never enter this rule."""
    if profile == "recall":
        return max(range(len(results)), key=lambda i: (results[i]["validation"]["f2"],
                   results[i]["validation"]["recall"], results[i]["validation"]["pr_auc"]))
    return max(range(len(results)), key=lambda i: (results[i]["validation"]["pr_auc"], results[i]["validation"]["f2"]))


def resampling_counts(model, labels):
    before = {str(label): int(count) for label, count in pd.Series(labels).value_counts().items()}
    after = before.copy()
    sampler = model.named_steps["sampler"]
    if isinstance(sampler, (SMOTE, RandomOverSampler)):
        for label, added in sampler.sampling_strategy_.items():
            after[str(label)] += int(added)
    elif isinstance(sampler, RandomUnderSampler):
        after.update({str(label): int(count) for label, count in sampler.sampling_strategy_.items()})
    return {"before": before, "after": after}


def training_folds(frame, target, strategy, seed=42):
    if strategy == "chronological":
        folds = []
        for train_indices, validation_indices in TimeSeriesSplit(n_splits=3).split(frame):
            # Minute-resolution timestamps can tie at a boundary. Exclude tied training rows.
            train_indices = train_indices[frame.Time.iloc[train_indices].to_numpy() < frame.Time.iloc[validation_indices[0]]]
            folds.append((train_indices, validation_indices))
    else:
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(frame, frame[target]))
    for train_indices, validation_indices in folds:
        if any(set(frame[target].iloc[indices].unique()) != {0, 1} for indices in [train_indices, validation_indices]):
            raise ValueError("Every CV training and validation fold must contain both fraud classes.")
    return folds


def train(csv_path: Path, output: Path, target=None, seed=42, tuning_rows=60000, eda=True, profile_name="recall",
          minimum_recall=None, candidate_names=None, split_strategy=None):
    if minimum_recall is not None and not 0 <= minimum_recall <= 1:
        raise ValueError("minimum_recall must be between 0 and 1.")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    frame, target, source = load_training_frame(csv_path, target)
    split_strategy = split_strategy or source["default_split"]
    if source.get("adapter") == "huggingface-transactions-v1" and profile_name == "recall" and candidate_names is None:
        candidate_names = ["XGBoost", "XGBoost · random oversampling"]
    profile = inspect_dataset(frame, target)
    profile["split_strategy"] = split_strategy
    print("DATASET", json.dumps(profile), flush=True)
    train_frame, validation, test = split_dataset(frame, target, seed, split_strategy)
    del frame
    schema = make_schema(train_frame, target)
    if source.get("adapter") == "huggingface-transactions-v1":
        labels = {"Amount": "Transaction amount (dataset units)", "Time": "Elapsed seconds since 2019-01-01",
                  "category": "Merchant category", "state": "State", "city_pop": "City population",
                  "lat": "Cardholder latitude", "long": "Cardholder longitude",
                  "merch_lat": "Merchant latitude", "merch_long": "Merchant longitude"}
        for feature in schema:
            feature["label"] = labels.get(feature["name"], feature["name"])
    names = [f["name"] for f in schema]
    partitions = {"train": train_frame, "validation": validation, "test": test}
    splits = {key: {"rows": len(part), "fraud": int(part[target].sum()), "legitimate": int((part[target] == 0).sum())}
              for key, part in partitions.items()}
    if "Time" in train_frame:
        for key, part in partitions.items():
            splits[key].update(time_min=float(part.Time.min()), time_max=float(part.Time.max()))
    for part in partitions.values():
        part.loc[:, names] = validate_features(part[names], schema)
    processed = Path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    write_json(processed / "split_indices.json", {key: part.index.tolist() for key, part in partitions.items()})
    if eda:
        create_eda(train_frame, profile, Path("reports/eda"), target)
    tuning = train_frame
    if len(tuning) > tuning_rows:
        tuning, _ = train_test_split(tuning, train_size=tuning_rows, stratify=tuning[target], random_state=seed)
    if split_strategy == "chronological":
        tuning = tuning.sort_values("Time", kind="stable")
    results, fitted = [], []
    cv = training_folds(tuning, target, split_strategy, seed)
    choices = candidates(schema, seed, profile_name)
    if candidate_names:
        unknown = set(candidate_names) - {choice[0] for choice in choices}
        if unknown:
            raise ValueError(f"Unknown candidates: {sorted(unknown)}")
        choices = [choice for choice in choices if choice[0] in candidate_names]
    for name, classifier, sampler, grid, imbalance in choices:
        print(f"TRAINING {name}: 3-fold {split_strategy} training CV, then full training partition", flush=True)
        begin = time.monotonic()
        pipe = Pipeline(preprocessing(schema).steps + [("sampler", sampler), ("classifier", classifier)])
        search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=1, error_score="raise", refit=False)
        search.fit(tuning[names], tuning[target])
        model = clone(pipe).set_params(**search.best_params_).fit(train_frame[names], train_frame[target])
        probabilities = model.predict_proba(validation[names])[:, 1]
        threshold = optimize_threshold(validation[target], probabilities, minimum_recall)
        entry = {"name": name, "imbalance_strategy": imbalance, "parameters": search.best_params_,
                 "cv_pr_auc": float(search.best_score_), "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                 "cv_trials": [{"parameters": params, "mean_pr_auc": float(mean), "std_pr_auc": float(std)} for params, mean, std in
                               zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])],
                 "threshold": threshold, "validation": evaluate(validation[target], probabilities, threshold),
                 "training_class_counts": resampling_counts(model, train_frame[target]),
                 "training_seconds": round(time.monotonic() - begin, 2)}
        results.append(entry)
        fitted.append((model, probabilities))
        print("VALIDATED", json.dumps(entry), flush=True)
    selected = select_candidate(results, profile_name)
    winner, val_probabilities = fitted[selected]
    threshold = results[selected]["threshold"]
    # Freeze selection before any test evaluation. Never refit on validation/test.
    print("FROZEN_SELECTION", results[selected]["name"], threshold, flush=True)
    for result, (model, _) in zip(results, fitted):
        result["test"] = evaluate(test[target], model.predict_proba(test[names])[:, 1], result["threshold"])
        result["selected"] = result is results[selected]
    probabilities = winner.predict_proba(test[names])[:, 1]
    prefix = "hf" if source.get("adapter") == "huggingface-transactions-v1" else "ulb"
    version = datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%dT%H%M%SZ")
    with csv_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    fitted_preprocessor = SklearnPipeline(winner.steps[:2])
    transformed_names = fitted_preprocessor.get_feature_names_out().tolist()
    classifier = winner.named_steps["classifier"]
    importance = getattr(classifier, "feature_importances_", None)
    importance_method = "model-native feature importance (global; not causal)"
    if importance is None and hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_[0])
        importance_method = "absolute coefficients in transformed feature space (global; scale dependent)"
    if importance is None:
        from sklearn.inspection import permutation_importance
        subset, _ = train_test_split(validation, train_size=min(15000, len(validation) - 1), stratify=validation[target], random_state=seed)
        importance = permutation_importance(winner, subset[names], subset[target], scoring="average_precision", n_repeats=2, random_state=seed, n_jobs=1).importances_mean
        transformed_names = names
        importance_method = "validation permutation importance: mean decrease in average precision over 2 repeats"
    feature_importance = sorted([{"feature": str(n), "importance": float(v)} for n, v in zip(transformed_names, importance)], key=lambda item: abs(item["importance"]), reverse=True)
    metadata = {
        "model_name": results[selected]["name"], "model_version": version, "training_date": datetime.now(timezone.utc).isoformat(),
        "features": schema, "selected_threshold": threshold, "risk_bands": {"medium": 0.3, "high": 0.7},
        "metrics": results[selected]["test"], "validation_metrics": results[selected]["validation"],
        "dataset": {**source, **profile, "sha256": digest, "file": csv_path.name, "splits": splits},
        "comparison": results, "curves": curves(test[target], probabilities),
        "threshold_curve": [{"threshold": float(t), **evaluate(validation[target], val_probabilities, float(t))}
                            for t in sorted(set(np.linspace(.01, .99, 50).tolist() + [threshold]))],
        "feature_importance": feature_importance, "importance_method": importance_method,
        "selection_metric": "f2" if profile_name == "recall" else "pr_auc",
        "minimum_validation_recall": minimum_recall,
        "imbalance_strategy": results[selected]["imbalance_strategy"],
        "training_class_counts": results[selected]["training_class_counts"],
        "selection_rule": ("Highest validation F2 (beta=2) among requested candidates; recall then average precision break ties. " if profile_name == "recall" else "Highest validation average precision (PR-AUC); F2 tie-break. ") + "CV tunes parameters by average precision. Threshold maximizes validation F2" + (f" subject to at least {minimum_recall:.0%} validation recall" if minimum_recall is not None else "") + ". No test metrics enter selection.",
        "probability_note": "Model scores are not calibrated probabilities of real-world fraud. Class weighting/resampling and distribution shift affect calibration.",
        "training": {"seed": seed, "profile": profile_name, "requested_candidates": candidate_names, "cv_folds": 3,
                     "cv_strategy": "expanding chronological windows" if split_strategy == "chronological" else "stratified shuffled folds",
                     "tuning_rows": len(tuning), "full_training_rows": len(train_frame),
                     "duration_seconds": round(time.monotonic() - started, 2), "duplicates_removed": profile["rows"] - sum(s["rows"] for s in splits.values())},
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "joblib"]},
    }
    predictions = test[names].copy()
    predictions["actual_class"] = test[target]
    predictions["fraud_probability"] = probabilities
    predictions["prediction"] = np.where(probabilities >= threshold, "Fraud", "Legitimate")
    predictions["risk_level"] = np.where(probabilities >= .7, "High", np.where(probabilities >= .3, "Medium", "Low"))
    predictions.to_csv(output / "test_predictions.csv", index=False)
    totals = {"total": len(test), "fraud": int((probabilities >= threshold).sum()), "legitimate": int((probabilities < threshold).sum()),
              "actual_fraud": int(test[target].sum()), "fraud_percentage": float((probabilities >= threshold).mean() * 100)}
    amounts = test["Amount"] if "Amount" in test else pd.Series([], dtype=float)
    recent = predictions.head(12).reset_index().rename(columns={"index": "source_row"}).to_dict(orient="records")
    timeline = []
    if "Time" in test:
        bucket_seconds = 86400 if source.get("time_origin") else 3600
        buckets = pd.DataFrame({"bucket": (test.Time // bucket_seconds).astype(int), "flagged": probabilities >= threshold})
        for bucket, group in buckets.groupby("bucket"):
            label = (pd.Timestamp(source["time_origin"]) + pd.Timedelta(seconds=int(bucket) * bucket_seconds)).strftime("%Y-%m-%d") if source.get("time_origin") else f"{bucket}h"
            timeline.append({"hour": int(bucket * bucket_seconds // 3600), "label": label, "transactions": len(group), "flagged": int(group.flagged.sum())})
    write_json(output / "evaluation_snapshot.json", {"scope": "held-out test set", "summary": totals,
               "probability_distribution": histogram(probabilities, np.linspace(0, 1, 11)),
               "amount_distribution": histogram(amounts, [0, 10, 25, 50, 100, 250, 500, 1000, max(1001, float(amounts.max()))]) if len(amounts) else [],
               "risk_distribution": [{"label": risk, "count": int((predictions.risk_level == risk).sum())} for risk in ["Low", "Medium", "High"]],
               "timeline": timeline, "recent": recent})
    # Examples come from training, not test labels. They are identified as benchmark examples in the UI.
    examples = {"legitimate": train_frame[train_frame[target] == 0][names].iloc[0].to_dict(),
                "fraud": train_frame[train_frame[target] == 1][names].iloc[0].to_dict()}
    write_json(output / "examples.json", examples)
    train_frame[names].sample(min(100, len(train_frame)), random_state=seed).to_csv(output / "sample_transactions.csv", index=False)
    joblib.dump(winner, output / "fraud_model.joblib", compress=3)
    joblib.dump(fitted_preprocessor, output / "preprocessor.joblib", compress=3)
    write_json(output / "model_metadata.json", metadata)
    filenames = ["fraud_model.joblib", "preprocessor.joblib", "model_metadata.json", "evaluation_snapshot.json", "examples.json", "sample_transactions.csv"]
    write_json(output / "manifest.json", {"model_version": version, "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in filenames}})
    comparison = pd.DataFrame([{"model": r["name"], "selected": r["selected"], "cv_pr_auc": r["cv_pr_auc"],
                               "validation_pr_auc": r["validation"]["pr_auc"], "validation_recall": r["validation"]["recall"],
                               "validation_precision": r["validation"]["precision"], "validation_f2": r["validation"]["f2"], "threshold": r["threshold"],
                               **{f"test_{k}": v for k, v in r["test"].items() if k != "confusion_matrix"}} for r in results])
    Path("reports").mkdir(exist_ok=True)
    comparison.to_csv("reports/model_comparison.csv", index=False)
    print("COMPLETE", json.dumps({"model": metadata["model_name"], "threshold": threshold, "test": metadata["metrics"], "seconds": metadata["training"]["duration_seconds"]}), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/hf_creditcard.csv"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--target", default=None, help="Default: is_fraud for the Hugging Face schema; Class for ULB")
    parser.add_argument("--split", choices=["chronological", "stratified"], default=None,
                        help="Default: chronological for Hugging Face; stratified for ULB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuning-rows", type=int, default=60000)
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--profile", choices=["recall", "benchmark"], default="recall",
                        help="recall: tree oversampling and validation-F2 selection; benchmark: original six AP comparisons")
    parser.add_argument("--minimum-recall", type=float, default=None,
                        help="Optional validation recall floor; default maximizes F2 without a recall constraint")
    parser.add_argument("--candidate", action="append", default=None,
                        help="Optional exact candidate name, repeat to select multiple; default evaluates the complete profile")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        train(args.data, args.output, args.target, args.seed, args.tuning_rows, not args.skip_eda, args.profile,
              args.minimum_recall, args.candidate, args.split)
