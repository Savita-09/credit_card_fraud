# Hugging Face model evaluation

Active model: **XGBoost · random oversampling**. Version: hf-20260910T092649Z.

Source: [santosh3110/credit_card_fraud_transactions](https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions). Revision: 63e73d05a06e1b87ef7dd9c6092eabc46245fcac.

## Dataset and split

1,048,575 source transactions; 6,006 fraud labels (0.5728%). All rows were used. The original CSV has 23 columns; the model accepts 9 input features.

| Partition | Rows | Fraud | Source time window |
|---|---:|---:|---|
| Train | 629,145 | 3,712 | 2019-01-01 00:00 to 2019-09-24 06:15 |
| Validation | 209,715 | 1,149 | 2019-09-24 06:16 to 2019-12-13 08:26 |
| Test | 209,715 | 1,145 | 2019-12-13 08:27 to 2020-03-10 16:08 |

Splits are chronological, with equal timestamps kept together. Each CV training window precedes its validation window. Customers may recur across time windows; this does not test generalization to entirely new customers.

## Measured comparison

| Candidate | Validation recall | Validation precision | Validation F2 | Test recall | Test precision | Test AP |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 78.5030% | 67.8195% | 0.761053 | 78.2533% | 64.7867% | 0.796087 |
| XGBoost · random oversampling | 81.8973% | 73.9780% | 0.801806 | 79.3013% | 69.3130% | 0.804547 |

Highest validation F2 (beta=2) among requested candidates; recall then average precision break ties. CV tunes parameters by average precision. Threshold maximizes validation F2. No test metrics enter selection.

The selected model uses a 0.25 minority/majority ratio: fraud training rows increase from 3,712 to 156,358; legitimate rows remain 625,433. Sampling runs only in training folds. Random oversampling preserves valid categorical combinations. No additional class weighting is applied.

The selected threshold is **0.7957711219787598**. Test confusion counts: TN 208,168, FP 402, FN 237, TP 908.

Oversampling, hyperparameter tuning and validation threshold selection jointly affect this comparison; it does not isolate a causal effect of sampling alone. The older ULB results use a different dataset and must not be compared as a before/after performance test.

## Reproducibility and limits

CSV SHA-256: 26b3b9c382c753e6cac75fb95a5ec9efe6832937a4e3fd777e6471c53a6eb490.

Public benchmark. The dataset card does not document its collection methodology.

The dataset card lists Apache-2.0. Currency and timezone are not specified, so the app displays dataset units and source-relative time. Names, card numbers, dates of birth, record identifiers and raw unix_time are excluded. Scores are uncalibrated and model flags are not confirmed fraud.

Artifact hashes, all 209,715 test predictions, saved-preprocessor parity and chronological partition boundaries were verified. See [verification details](huggingface_verification.json).
