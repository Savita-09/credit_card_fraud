# Dataset and EDA findings

The source fraud prevalence is 0.5728%; always predicting legitimate would look accurate.

0 exact duplicate rows are removed before splitting to avoid overlap.

No outlier rows are discarded: rare extremes can be useful fraud signals. Robust scaling and log amount reduce scale sensitivity.

Transaction category, amount and location signals are modeled. Names, card numbers and record identifiers are excluded.

Time is elapsed seconds from the configured source origin, not a verified local timezone.

Chronological train/validation/test windows evaluate later transactions; timestamps tied at boundaries stay together. Customers can recur across windows, so this is not an unseen-customer benchmark.

All distribution and correlation charts use only the training partition.
