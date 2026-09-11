"""Oversampling must change training prevalence, never validation/inference prevalence."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_processing import make_schema
from src.feature_engineering import preprocessing
from src.train import candidates, resampling_counts, select_candidate
from src.evaluate import evaluate, optimize_threshold


class RecordingLogistic(LogisticRegression):
    def fit(self, X, y, **kwargs):
        self.training_counts_ = pd.Series(y).value_counts().to_dict()
        return super().fit(X, y, **kwargs)


def test_cv_oversamples_only_training_fold():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"Amount": rng.uniform(0, 100, 330), "V1": rng.normal(size=330),
                          "Class": np.r_[np.zeros(300, dtype=int), np.ones(30, dtype=int)]})
    schema = make_schema(frame, "Class")
    X, y = frame.drop(columns="Class"), frame.Class
    pipeline = Pipeline(preprocessing(schema).steps + [
        ("sampler", SMOTE(sampling_strategy=.5, k_neighbors=3, random_state=42)),
        ("classifier", RecordingLogistic(max_iter=300)),
    ])
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y))
    result = cross_validate(pipeline, X, y, cv=folds, scoring="average_precision", return_estimator=True)
    for estimator, (train_indices, validation_indices) in zip(result["estimator"], folds):
        assert estimator.named_steps["classifier"].training_counts_ == {0: 200, 1: 100}
        # The imputer is fitted before oversampling and never sees held-out rows.
        actual_median = estimator.named_steps["columns"].named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
        assert actual_median == X.iloc[train_indices].Amount.median()
        untouched = X.iloc[validation_indices].copy(deep=True)
        scores = estimator.predict_proba(untouched)
        assert scores.shape == (110, 2)
        pd.testing.assert_frame_equal(untouched, X.iloc[validation_indices])
        assert resampling_counts(estimator, y.iloc[train_indices]) == {"before": {"0": 200, "1": 20}, "after": {"0": 200, "1": 100}}


def test_recall_profile_oversamples_tree_models_without_double_weighting():
    schema = [{"name": "V1", "type": "number"}]
    choices = candidates(schema, 42, "recall")
    oversampled = [(name, estimator, sampler, grid) for name, estimator, sampler, grid, _ in choices if isinstance(sampler, (SMOTE, RandomOverSampler))]
    assert len(oversampled) == 3
    assert any("Random Forest" in name and isinstance(sampler, SMOTE) for name, _, sampler, _ in oversampled)
    for _, estimator, _, grid in oversampled:
        assert estimator.get_params().get("class_weight") is None
        assert grid["sampler__sampling_strategy"] == [.1, .25]


def test_mixed_categories_use_random_oversampling_not_interpolated_one_hot():
    choices = candidates([{"name": "merchant", "type": "category"}], 42, "recall")
    assert any(isinstance(sampler, RandomOverSampler) for _, _, sampler, _, _ in choices)
    assert not any(isinstance(sampler, SMOTE) for _, _, sampler, _, _ in choices)


def test_selection_prioritizes_validation_f2_and_ignores_test_metrics():
    results = [
        {"validation": {"pr_auc": .95, "f2": .75, "recall": .7}, "test": {"recall": 1.}},
        {"validation": {"pr_auc": .85, "f2": .85, "recall": .9}, "test": {"recall": 0.}},
    ]
    assert select_candidate(results, "recall") == 1
    assert select_candidate(results, "benchmark") == 0


def test_recall_floor_catches_fraud_missed_by_unconstrained_f2():
    labels = np.r_[np.ones(10), np.zeros(100)]
    scores = np.r_[np.linspace(.8, 1., 8), .2, .1, np.linspace(.21, .7, 100)]
    unconstrained = optimize_threshold(labels, scores)
    assert evaluate(labels, scores, unconstrained)["recall"] == .8
    threshold = optimize_threshold(labels, scores, minimum_recall=.9)
    metrics = evaluate(labels, scores, threshold)
    assert metrics["recall"] >= .9
    feasible_f2 = [evaluate(labels, scores, t)["f2"] for t in scores
                   if evaluate(labels, scores, t)["recall"] >= .9]
    assert metrics["f2"] >= max(feasible_f2) - 1e-12
