# Verification record · 10 September 2026

The real ULB pipeline was trained and evaluated. Model version: `ulb-20260910T074013Z`.

| Check | Result |
|---|---|
| Raw-source inspection | 284,807 rows; 492 fraud; no missing values; 1,081 duplicates |
| CV and training | Six candidate/imbalance combinations; 3 folds × 2 parameter choices; full training refit |
| Frozen selection | Random Forest; validation AP 0.7929948529; threshold 0.2877936236332483 |
| Held-out test | AP 0.8347952579; precision 0.9156626506; recall 0.8; F1 0.8539325843 |
| Confusion matrix | TN 56,644 / FP 7 / FN 19 / TP 76 |
| Python/API/ML tests | 41 passed; two upstream TestClient deprecation warnings |
| Frontend tests | 4 passed |
| Frontend production build | Passed; separate application, React and chart chunks |
| EDA notebook | Executed all 5 code cells; no errors |
| Preprocessing notebook | Executed all 4 code cells; no errors |
| Training/evaluation notebook | Executed all 5 code cells; no errors |
| Offline CSV prediction | Scored and exported 100 sample transactions |
| Actual API startup | `/health` healthy; `/docs` HTTP 200 |
| Actual frontend startup | HTTP 200; same-origin `/api/health` healthy |
| Browser single prediction | Training fraud example displayed a 63.96% score; Fraud flag; Medium risk; six sensitivity signals |
| Browser batch | Three training-example rows validated and scored; two legitimate and one fraud flag |
| Browser filter | Fraud filter displayed the single flagged result |
| Browser CSV download | Downloaded file parsed successfully: three rows, correct decisions and five score/metadata columns |
| Live history after restart | Four analyses retained: two flags and two legitimate decisions from browser verification |
| Browser layouts | Desktop 1280 px, tablet 820 px and mobile 390 px; no page-width overflow; mobile navigation worked |
| Browser console | No errors or warnings in final desktop reload |
| Compose syntax | Parsed; backend/frontend services, ports, health dependencies and persistence configured |
| Docker build/start | **Not executed: Docker is not installed on this host** |

Commands used included `python -m pytest`, `npm test`, `npm run build`, `python -m scripts.build_notebooks`, and `python -m src.predict models/sample_transactions.csv --output .tmp/offline_predictions.csv`. Windows sandbox restrictions required approved unsandboxed execution for worker synchronization, Jupyter kernel-file permissions and Vite child processes. The final commands passed in the normal host environment.

Test fixtures are synthetic only for testing API/transform contracts; displayed benchmark metrics come from the actual ULB run. The live records left by browser testing use publicly available anonymized training examples and are not independent performance evidence.

Screenshots are under `docs/screenshots/`. The primary app remains available at `http://localhost:5173` with FastAPI at `http://localhost:8000`. Development process IDs and logs are in `.tmp/`.
