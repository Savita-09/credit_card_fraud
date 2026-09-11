# Dataset and EDA findings

The source fraud prevalence is 0.1727%; always predicting legitimate would look accurate.

1,081 exact duplicate rows are removed before splitting to avoid overlap.

No outlier rows are discarded: rare extremes can be useful fraud signals. Robust scaling and log amount reduce scale sensitivity.

V1–V28 are anonymized PCA coordinates; their business meanings cannot be recovered from this dataset.

Time is elapsed seconds from the first transaction, not a date or verified local time.

Random stratification is a benchmark. A future temporal evaluation is needed before deployment on changing transaction patterns.

All distribution and correlation charts use only the training partition.
