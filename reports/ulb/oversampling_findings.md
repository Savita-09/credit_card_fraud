# Oversampling results and deployment decision

SMOTE and random oversampling are implemented and tested in the training pipeline. The evaluated SMOTE candidate is saved, but **the active Random Forest was not replaced**: the candidate's precision deteriorated on validation data.

## What changed

- Training-only imbalanced-learn pipelines compare weighted Random Forest, Random Forest with duplicated fraud rows, Random Forest with SMOTE, and XGBoost with SMOTE.
- Three-fold CV tunes model depth and minority/majority ratios of 0.10 and 0.25. Imputation and scaling fit within the training fold before sampling. Validation and test rows are never resampled.
- Oversampled classifiers do not also apply class weights. Mixed categorical inputs use random oversampling instead of interpolating one-hot values.
- The recall profile selects by validation F2; optional `--minimum-recall` constrains threshold selection. The default leaves this constraint disabled because aggressive targets produced many false alerts.
- Fitted class counts, selection criteria and thresholds are saved in the model metadata and supported by the dashboard.

## Measured candidate

The initial unconstrained comparison did not improve validation recall: the weighted baseline reached 78.72%; all three oversampled candidates reached 77.66% at their F2-optimal thresholds. See [initial comparison](oversampling_initial_comparison.json).

XGBoost with SMOTE had the highest validation average precision among the oversampled candidates. Its CV-selected configuration uses depth 5, 220 trees and a 0.10 minority/majority ratio. Fraud training examples increase from **284 to 16,995**; legitimate examples remain 169,951. A validation recall floor of 0.80 selects threshold **0.05792403593659401**.

| Original test partition | Active Random Forest | SMOTE candidate |
|---|---:|---:|
| Fraud recall | 80.00% | 89.47% |
| Fraud precision | 91.57% | 36.48% |
| Detected fraud | 76 | 85 |
| Missed fraud | 19 | 10 |
| False alerts | 7 | 148 |
| F2 | 0.8207 | 0.6933 |
| Average precision | 0.8348 | 0.8547 |

The apparent test recall gain combines a different classifier, oversampling and a lower decision threshold. It does not establish that oversampling alone caused the improvement.

## Comparison at the same validation recall

Lowering the old model's threshold also improves recall. At an 80% validation recall floor:

| Validation result | Existing RF with retuned threshold | SMOTE candidate |
|---|---:|---:|
| Threshold | 0.10907080221687897 | 0.05792403593659401 |
| Recall | 80.85% | 80.85% |
| Precision | 67.26% | 34.70% |
| Detected fraud | 76 | 76 |
| False alerts | 37 | 143 |

That comparison favors retaining the existing model. **Neither this alternative RF threshold nor the SMOTE candidate was activated.** The active threshold remains 0.2877936236332483. A 90% validation recall floor for SMOTE required 4,648 validation false alerts, so it was not adopted as a default.

These are exploratory results on the same benchmark partitions used earlier, not a new independent evaluation. There are only 94 validation fraud cases and 95 test fraud cases. Scores are uncalibrated. A higher recall target requires an explicit choice of acceptable review workload; oversampling does not guarantee a better trade-off.

## Reproduce and inspect

The evaluated bundle is in `artifacts/real-data/ulb-20260910T085138Z/`, including the full fitted SMOTE pipeline, preprocessor, metadata, predictions and integrity manifest. The original bundle remains in `models/` and its backup in `artifacts/real-data/ulb-20260910T074013Z/`.

```bash
python -m src.train --profile recall --candidate "XGBoost · SMOTE" --minimum-recall 0.80 --data data/raw/creditcard.csv --output artifacts/experiments/smote
```

Omit `--candidate` to compare all four candidates. Omit `--minimum-recall` to use unconstrained validation F2. Training writes the comparison CSV to `reports/model_comparison.csv`; the retained experiment CSV is [oversampling_model_comparison.csv](oversampling_model_comparison.csv).

See [full before/after metrics](oversampling_comparison.md), [baseline threshold alternatives](baseline_recall_targets.json) and [SMOTE threshold alternatives](smote_recall_targets.json).

Validation: 46 backend/ML tests and 4 frontend tests passed; the production frontend build passed. Artifact inference verification and executed notebooks accompany these results.
