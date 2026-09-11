# Sentinel · Fraud Intelligence

Credit-card fraud detection using the Hugging Face dataset [santosh3110/credit_card_fraud_transactions](https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions), an XGBoost model with training-only random oversampling, FastAPI inference and a React dashboard.

The active saved model uses the Hugging Face dataset. Earlier ULB experiments are retained under reports/ulb/ and artifacts/real-data/ for reference.

## Run the application

Python 3.12 and Node 22+ are used by this project. The saved model is included in models/, so serving does not require downloading the dataset or retraining.

~~~bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
~~~

In a second terminal:

~~~bash
cd frontend
npm ci
npm run dev
~~~

On Windows, after dependencies are installed, scripts/start_local.ps1 starts both services in hidden background processes and records their PIDs/logs in .tmp/. It refuses to take over occupied ports.

- [Dashboard](http://localhost:5173)
- [API health](http://localhost:8000/health)
- [Interactive API documentation](http://localhost:8000/docs)

The application includes an evaluation/live overview, single-transaction scoring, CSV validation and batch scoring, CSV export, model comparison, confusion counts, ROC/PR curves, threshold controls, feature importance and local sensitivity signals.

## Dataset and inputs

The pinned Hugging Face revision contains **1,048,575 transactions**, including **6,006 fraud labels (0.5728%)**. The source CSV has 23 columns and uses is_fraud as the label. The minimal dataset card does not document collection methodology; this project treats it as a public benchmark rather than verified real-world payment traffic.

The model accepts nine features:

| Model input | Source / meaning |
|---|---|
| Amount | amt, in dataset units |
| Time | Seconds since 2019-01-01 00:00:00, derived from trans_date_trans_time |
| category | Merchant category |
| state | State |
| city_pop | City population |
| lat, long | Cardholder coordinates |
| merch_lat, merch_long | Merchant coordinates |

Names, card numbers, record identifiers, dates of birth and raw unix_time are excluded. Currency and timezone are not specified in the dataset card, so the app does not assume EUR or a verified local timezone. Read [source provenance and feature mapping](docs/dataset.md).

The earlier uploaded loan-data archive and ULB CSV are retained for reference. Neither is used by the active model.

## Download and retrain

~~~bash
python -m scripts.download_data
python -m src.train
python -m scripts.build_notebooks
~~~

The downloader pins revision 63e73d05a06e1b87ef7dd9c6092eabc46245fcac, verifies the published ZIP checksum, extracts the CSV into a fixed path and records its SHA-256 in data/raw/hf_creditcard.source.json. Cached data is verified before reuse. No Hugging Face credentials or dataset execution scripts are required.

Default training compares XGBoost with and without random oversampling. It uses the complete dataset, with a reproducible 60,000-row training subset for hyperparameter tuning and a full training-partition refit.

- Chronological 60/20/20 partitions: 629,145 training, 209,715 validation and 209,715 test rows.
- Three expanding chronological CV windows. Equal timestamps never appear on both sides of a training/validation boundary.
- Training-fitted median imputation, robust scaling and unknown-safe categorical encoding.
- Shared feature engineering: log amount, daily/weekly time cycles and great-circle merchant distance.
- Random oversampling inside each training fold, with minority/majority ratios 0.10 and 0.25 compared by CV. Complete rows are duplicated, preserving categorical combinations.
- No additional class weighting on oversampled classifiers.
- CV tunes model depth and sampling ratio by average precision. Validation F2 selects the model and threshold before test evaluation.
- Validation/test rows are never oversampled, and the selected model is not refitted on them.

An optional --minimum-recall adds a validation recall floor; it is disabled by default and is not a guarantee on test or future data. --candidate can select named candidates for an experiment. Numeric-only schemas also support SMOTE; categorical schemas use random oversampling.

The older source remains available with python -m scripts.download_data --dataset ulb and python -m src.train --data data/raw/creditcard.csv --profile benchmark.

## Measured results

Active run: **hf-20260910T092649Z**. Selected model: **XGBoost · random oversampling**.

| Later test window | Baseline XGBoost | Selected oversampled model |
|---|---:|---:|
| Fraud recall | 78.25% | **79.30%** |
| Fraud precision | 64.79% | **69.31%** |
| F2 | 0.7513 | **0.7708** |
| Average precision | 0.7961 | **0.8045** |
| Detected fraud | 896 | **908** |
| False alerts | 487 | **402** |

The selected threshold is **0.7957711219787598**. The chosen 0.25 sampling ratio increases fraud training examples from 3,712 to 156,358, with 625,433 legitimate training examples retained.

See the [measured model report](reports/model_report.md), [comparison CSV](reports/model_comparison.csv) and [completed verification](reports/verification.md). Oversampling, hyperparameter tuning and threshold selection jointly affect the comparison. The older ULB measurements use a different dataset and are not a before/after comparison with this run.

Customers may recur across time windows. Results therefore measure later transactions, not entirely unseen customers. Scores are uncalibrated, and model flags do not establish fraud or execute payment approvals/declines.

## Use raw Hugging Face rows in the app

The CSV upload expects the nine model input columns, with labels and identifiers removed. The preparation command performs the mapping and time conversion:

~~~bash
python -m scripts.prepare_transactions --input data/raw/hf_creditcard.csv --output data/processed/hf_demo.csv --limit 1000
~~~

Upload that prepared file on the Batch page. The Sample CSV button also downloads a ready-to-score file drawn from training rows.

All model feature columns are required; individual empty cells are imputed. Unknown category values are supported. The preparation command rejects invalid dates. The API rejects malformed CSV, unexpected columns, invalid numbers, out-of-range coordinates and negative amounts/times/populations. Default upload limits are 5 MiB and 10,000 rows.

## API and offline inference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Model readiness |
| GET | /model-info | Dataset, feature schema, versions, metrics and active settings |
| GET | /statistics?scope=evaluation | Frozen later-test evaluation |
| GET | /statistics?scope=live | Stored predictions for the active model |
| GET | /examples | Labeled training examples for demonstrations |
| GET | /sample-csv | Prepared training examples with labels removed |
| POST | /predict | Single JSON transaction |
| POST | /predict/batch/validate | CSV validation and preview |
| POST | /predict/batch | Batch results and CSV output |

~~~python
import httpx
base = "http://localhost:8000"
features = httpx.get(f"{base}/examples").json()["transactions"]["fraud"]
response = httpx.post(f"{base}/predict", json={"features": features, "explain": True})
response.raise_for_status()
print(response.json())
~~~

~~~bash
python -m src.predict models/sample_transactions.csv --output predictions.csv
curl -F "file=@models/sample_transactions.csv" "http://localhost:8000/predict/batch?format=csv" -o predictions.csv
~~~

Per-request thresholds affect new decisions, while evaluation metrics retain the validation-selected threshold. Low/Medium/High risk bands are separate from classification. Explanations replace one feature with its training median/mode and recompute all derived features; the resulting sensitivities are non-additive and non-causal.

SQLite history stores scores, decisions, amounts, IDs and model versions, rather than full input feature vectors. Earlier model history is retained but excluded from the new model's live view.

## Configuration, deployment and checks

requirements.txt is the shared Python dependency list for local setup, packaging and Docker. Copy .env.example to .env for optional configuration; environment variables use the FRAUD_ prefix. The default threshold comes from the saved model unless FRAUD_DEFAULT_THRESHOLD is set. Configure VITE_API_BASE_URL and the backend CORS allowlist when hosting them separately.

~~~bash
python -m pytest
cd frontend
npm test
npm run build
~~~

The test suite covers dataset adaptation, excluded identifiers, source checksums, chronological boundaries, oversampling leakage, preprocessing/inference parity, API validation, risk settings, upload limits, history, artifact tampering and frontend interactions.

~~~bash
docker compose up --build
~~~

Compose serves the UI on 5173 and API on 8000, binds ports to localhost and persists prediction history. Docker is not installed on this host, so the containers were not executed here. Local API inference, tests and the frontend build were verified. See [architecture](docs/architecture.md) before extending deployment.

## Project map

~~~text
backend/                  FastAPI, validation, inference and prediction history
frontend/                 React dashboard, forms, charts and frontend tests
src/dataset.py            Hugging Face source adapter and provenance checks
src/data_processing.py    Validation and chronological/stratified splitting
src/feature_engineering.py Shared transformation pipeline
src/train.py              CV, oversampling, selection and artifact persistence
src/evaluate.py           Metrics, curves and threshold selection
src/predict.py            Offline CSV inference
models/                   Active Hugging Face model and verified artifacts
artifacts/real-data/       Preserved earlier ULB model bundles
data/raw/                 Source CSVs and checksum manifests
notebooks/                Three executed notebooks for the active dataset
reports/                  Current model report, EDA and verification
reports/ulb/              Historical ULB results
tests/web/                Backend and ML tests
scripts/                  Download, preparation, notebook execution and startup
docs/                     Dataset provenance and architecture
~~~

The approved obsolete streaming/synthetic experiment, its tests/configuration and temporary copies were removed. Installed dependencies, active service logs/PIDs, source datasets, model backups and prediction history were preserved.
