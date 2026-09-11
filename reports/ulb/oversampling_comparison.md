# Oversampling and fraud recall

Baseline: Random Forest (ulb-20260910T074013Z).
Evaluated candidate: XGBoost · SMOTE (ulb-20260910T085138Z).

These are candidate evaluation results, not confirmation that the active model was replaced.

## Measured comparison

| Split / metric | Before | After |
|---|---:|---:|
| Validation recall | 0.787234 | 0.808511 |
| Validation precision | 0.913580 | 0.347032 |
| Validation f1 | 0.845714 | 0.485623 |
| Validation f2 | 0.809628 | 0.638655 |
| Validation pr_auc | 0.792995 | 0.794526 |
| Test recall | 0.800000 | 0.894737 |
| Test precision | 0.915663 | 0.364807 |
| Test f1 | 0.853933 | 0.518293 |
| Test f2 | 0.820734 | 0.693312 |
| Test pr_auc | 0.834795 | 0.854721 |

Test detected fraud: 76 -> 85; missed fraud: 19 -> 10; false positives: 7 -> 148.
Threshold: 0.287793623633 -> 0.057924035937.

## How the change works

Fraud training examples: 284 -> 16,995. Legitimate examples: 169,951 -> 169,951.
Numeric SMOTE runs after training-fitted imputation/scaling, inside each CV fold. Mixed categorical data uses random oversampling instead.
Oversampled models do not also apply class weights. CV selects between minority/majority ratios 0.10 and 0.25 and model-depth settings.
Highest validation F2 (beta=2) among requested candidates; recall then average precision break ties. CV tunes parameters by average precision. Threshold maximizes validation F2 subject to at least 80% validation recall. No test metrics enter selection.

## Interpretation

The change combines oversampling, parameter tuning, F2 model selection and threshold optimization. The overall comparison does not isolate the causal effect of oversampling alone.
Validation and test rows were never resampled. Thresholds were calculated from validation data. This exploratory follow-up reuses the earlier benchmark test set for comparability; it is not a new independent external evaluation.
Higher recall can mean more false alerts. Scores remain uncalibrated. The earlier artifact is preserved under artifacts/real-data/ for rollback.
