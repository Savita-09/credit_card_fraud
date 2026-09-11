# Dataset provenance and feature mapping

The active dataset is [santosh3110/credit_card_fraud_transactions](https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions). The download is pinned to revision **63e73d05a06e1b87ef7dd9c6092eabc46245fcac** and verified against the published archive SHA-256.

- Archive SHA-256: 0be20e0480cd79790f92e2c103acbd8b2c0500c7dd85fc4d12d67ac6b134548e
- Extracted CSV SHA-256: 26b3b9c382c753e6cac75fb95a5ec9efe6832937a4e3fd777e6471c53a6eb490
- Local CSV: data/raw/hf_creditcard.csv
- Source manifest: data/raw/hf_creditcard.source.json
- License listed in the dataset card: Apache-2.0
- Rows: 1,048,575; fraud labels: 6,006; prevalence: 0.5728%
- No missing modeled values and no duplicate modeled rows were found.
- Source date range: 2019-01-01 00:00 to 2020-03-10 16:08

Public benchmark. The dataset card does not document its collection methodology. Currency and timezone are not specified. The app therefore uses dataset units and source-relative timestamps; it does not claim these are verified real-world payment records.

## Raw columns to model inputs

| Source column | Model input | Treatment |
|---|---|---|
| amt | Amount | Nonnegative amount in source units |
| trans_date_trans_time | Time | Seconds since 2019-01-01 00:00:00, using month/day/year source dates |
| category | category | Categorical merchant category; unknown-safe encoding |
| state | state | Categorical state; unknown-safe encoding |
| city_pop | city_pop | Nonnegative population |
| lat, long | lat, long | Cardholder coordinates in degrees |
| merch_lat, merch_long | merch_lat, merch_long | Merchant coordinates in degrees |
| is_fraud | Training label only | 0 legitimate, 1 fraud; never a prediction input |

Excluded columns: Unnamed: 0, cc_num, merchant, first, last, gender, street, city, zip, job, dob, trans_num, unix_time. Card numbers are not parsed as numeric predictors, and merchant names/record identifiers cannot enter the model. Raw unix_time is excluded in favor of the human-readable source timestamp.

The shared pipeline adds log amount, daily/weekly cyclic time features and great-circle distance between cardholder and merchant coordinates. Median imputation, robust scaling and categorical encoding are fitted inside training folds. Random oversampling duplicates valid minority rows within training folds; validation and test prevalence are preserved.

## Evaluation

Chronological 60/20/20 windows contain 629,145, 209,715 and 209,715 rows respectively. Equal timestamps do not cross a boundary. Training uses expanding chronological CV windows on 60,000 sampled training rows, then full-partition refits. Model/threshold selection uses validation data, and test metrics are computed afterward. Customers may recur across partitions, so this is not an unseen-customer evaluation.

Read the [measured model report](../reports/model_report.md). Earlier ULB reports are archived under reports/ulb/ and their model bundles under artifacts/real-data/; those measurements describe a different dataset. The original uploaded loan-data archive and earlier ULB CSV are retained for reference and are not used by the active model.
