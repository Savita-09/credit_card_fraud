"""Compare validated oversampling artifacts with the saved pre-change model."""
import argparse
import json
from pathlib import Path

def compare(baseline_path, candidate_path, output):
    baseline = json.loads((baseline_path / "model_metadata.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_path / "model_metadata.json").read_text(encoding="utf-8"))
    if (baseline["dataset"]["sha256"] != candidate["dataset"]["sha256"]
            or baseline["dataset"]["splits"] != candidate["dataset"]["splits"]
            or baseline["training"]["seed"] != candidate["training"]["seed"]):
        raise ValueError("Models must use the same source, split seed and partition counts for this comparison.")
    changes = {}
    for partition, key in [("validation", "validation_metrics"), ("test", "metrics")]:
        changes[partition] = {"before": baseline[key], "after": candidate[key],
                              "recall_change_percentage_points": 100 * (candidate[key]["recall"] - baseline[key]["recall"])}
    report = {"baseline_version": baseline["model_version"], "candidate_version": candidate["model_version"],
              "model_name": candidate["model_name"], "selection_rule": candidate["selection_rule"],
              "training_class_counts": candidate["training_class_counts"],
              "threshold_before": baseline["selected_threshold"], "threshold_after": candidate["selected_threshold"],
              "changes": changes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Oversampling and fraud recall", "",
             f"Baseline: {baseline['model_name']} ({baseline['model_version']}).",
             f"Evaluated candidate: {candidate['model_name']} ({candidate['model_version']}).", "",
             "These are candidate evaluation results, not confirmation that the active model was replaced.", "",
             "## Measured comparison", "", "| Split / metric | Before | After |", "|---|---:|---:|"]
    for partition in ["validation", "test"]:
        for metric in ["recall", "precision", "f1", "f2", "pr_auc"]:
            lines.append(f"| {partition.title()} {metric} | {changes[partition]['before'][metric]:.6f} | {changes[partition]['after'][metric]:.6f} |")
    before, after = baseline["metrics"]["confusion_matrix"], candidate["metrics"]["confusion_matrix"]
    counts = candidate["training_class_counts"]
    lines.extend(["", f"Test detected fraud: {before[1][1]} -> {after[1][1]}; missed fraud: {before[1][0]} -> {after[1][0]}; false positives: {before[0][1]} -> {after[0][1]}.",
                  f"Threshold: {baseline['selected_threshold']:.12f} -> {candidate['selected_threshold']:.12f}.", "",
                  "## How the change works", "",
                  f"Fraud training examples: {counts['before']['1']:,} -> {counts['after']['1']:,}. Legitimate examples: {counts['before']['0']:,} -> {counts['after']['0']:,}.",
                  "Numeric SMOTE runs after training-fitted imputation/scaling, inside each CV fold. Mixed categorical data uses random oversampling instead.",
                  "Oversampled models do not also apply class weights. CV selects between minority/majority ratios 0.10 and 0.25 and model-depth settings.",
                  candidate["selection_rule"], "",
                  "## Interpretation", "",
                  "The change combines oversampling, parameter tuning, F2 model selection and threshold optimization. The overall comparison does not isolate the causal effect of oversampling alone.",
                  "Validation and test rows were never resampled. Thresholds were calculated from validation data. This exploratory follow-up reuses the earlier benchmark test set for comparability; it is not a new independent external evaluation.",
                  "Higher recall can mean more false alerts. Scores remain uncalibrated. The earlier artifact is preserved under artifacts/real-data/ for rollback.", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/oversampling_comparison.md"))
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.output)
