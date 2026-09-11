# Hugging Face integration verification

Verified on 2026-09-10 with active model `hf-20260910T092649Z`.

## Data and model

- Source: [santosh3110/credit_card_fraud_transactions](https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions), pinned revision `63e73d05a06e1b87ef7dd9c6092eabc46245fcac`.
- Published archive SHA-256 and extracted CSV SHA-256 verified. All 1,048,575 source rows were used in chronological train/validation/test partitions.
- Oversampling is fitted only on training rows within CV and final training. Validation and test prevalence remain unchanged. Equal timestamps do not straddle split/CV boundaries.
- Verified the saved artifact manifest, package compatibility, standalone preprocessing parity, and all 209,715 test prediction scores against the active model. Recomputed confusion matrix: `[[208168, 402], [237, 908]]`.
- Selected XGBoost with random oversampling has test fraud recall 79.30% and precision 69.31%, versus baseline XGBoost recall 78.25% and precision 64.79% on the same test window. See [model report](model_report.md) and [artifact checks](huggingface_verification.json).

## Automated checks

| Check | Result |
|---|---|
| `python -m pytest --basetemp=.tmp/huggingface-final-tests` | 50 passed; two upstream deprecation warnings |
| `npm test` in frontend | 5 passed |
| `npm run build` in frontend | Passed |
| `python -m scripts.build_notebooks` | All three notebooks executed; 14 code cells, no error outputs |
| Preparation CLI on 1,000 source rows | Nine model features, identifiers and label excluded; API validation passed |
| Offline sample CSV inference | All 100 scores, decisions, thresholds and model versions match HTTP batch inference |

Notebook details are recorded in [notebook verification](notebook_verification.json).

## Running application

- `/health` reports healthy. `/model-info` reports the Hugging Face source and selected threshold `0.7957711219787598` as the active threshold.
- Single-transaction HTTP inference returns a fraud flag with score 0.987108 for the labeled fraud training example, with sensitivity signals.
- A three-row CSV validates and scores as one fraud flag and two legitimate predictions. Sample CSV download and CSV results parse successfully.
- Browser inspection confirmed the active model and metrics on the dashboard and performance page, dataset units without an assumed currency, source date labels, and the nine named transaction inputs.
- Browser example loading and submission returned the same 98.71% fraud score and active model version. Obsolete anonymized-feature text was removed and threshold help now reflects whether a validation recall floor is configured.
- API smoke checks are recorded in [API verification](huggingface_api_verification.json). Smoke tests added public benchmark demonstrations to live history; these are not independent evaluation results. Previous model history was preserved and remains filtered by model version.

## Approved cleanup

The approved legacy cleanup removed 56 targets containing 365 files (123.39 MiB): the obsolete synthetic/streaming app, its tests and configuration, synthetic data/database, legacy artifacts, MLflow runs and temporary copies. The [cleanup manifest](legacy_cleanup.json) records the paths. New test and notebook caches may be recreated when checks run.

The active app, installed dependencies, datasets, original uploaded archive, earlier model bundles, historical oversampling reports, prediction history and active service logs/PIDs were preserved. Earlier ULB reports are under `reports/ulb/`; they use a different dataset and are not directly comparable with this run.

## Limits

Docker was unavailable on this host, so containers were not executed. This source's card does not establish collection methodology, currency or timezone. The chronological benchmark can include recurring customers across windows and is not an unseen-customer evaluation. Scores remain uncalibrated; validation/test improvements do not guarantee future recall.
