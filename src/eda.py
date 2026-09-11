"""Reproducible EDA outputs. Distribution analysis uses the training partition only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_eda(train: pd.DataFrame, profile: dict, output: Path, target="Class"):
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=["#7357dd", "#e26b79"])
    plots = []

    def save(name):
        plt.tight_layout()
        plt.savefig(output / name, dpi=130, bbox_inches="tight")
        plt.close()
        plots.append(name)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = train[target].value_counts().sort_index()
    ax.bar(["Legitimate", "Fraud"], counts, color=["#7357dd", "#e26b79"])
    ax.set_yscale("log")
    ax.set(title="Training class distribution", ylabel="Transactions (log scale)")
    for i, count in enumerate(counts):
        ax.text(i, count * 1.05, f"{count:,}", ha="center")
    save("class_distribution.png")

    amount_summary = {}
    if "Amount" in train:
        amount_summary = train.groupby(target).Amount.describe().to_dict()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(data=train, x="Amount", bins=60, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_title("Amount distribution · log count")
        view = train.assign(log_amount=np.log1p(train.Amount))
        sns.histplot(data=view, x="log_amount", hue=target, stat="density", common_norm=False,
                     element="step", bins=45, ax=axes[1])
        axes[1].set_title("Amount by class · log(1 + amount)")
        sns.boxplot(data=view, x=target, y="log_amount", ax=axes[2])
        axes[2].set_title("Amount outliers retained")
        save("amount_distributions.png")
    if "Time" in train:
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.histplot(data=train, x="Time", hue=target, bins=48, stat="density", common_norm=False,
                     element="step", ax=ax)
        ax.set_title("Elapsed transaction time by class (training only)")
        save("time_distribution.png")
    numeric = train.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Training Pearson correlations · association is not causation")
    save("correlations.png")
    # Show every numeric feature, not only the ones most associated with the label.
    columns = [c for c in numeric if c != target]
    for start in range(0, len(columns), 12):
        subset = columns[start:start + 12]
        fig, axes = plt.subplots(int(np.ceil(len(subset) / 3)), 3, figsize=(14, 3 * int(np.ceil(len(subset) / 3))))
        for ax, name in zip(np.asarray(axes).ravel(), subset):
            sns.histplot(data=train, x=name, hue=target, bins=40, stat="density", common_norm=False,
                         element="step", legend=False, ax=ax)
            ax.set_yscale("symlog", linthresh=0.001)
        for ax in np.asarray(axes).ravel()[len(subset):]:
            ax.set_visible(False)
        save(f"feature_distributions_{start + 1}.png")
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    outliers = ((numeric < q1 - 1.5 * (q3 - q1)) | (numeric > q3 + 1.5 * (q3 - q1))).sum().to_dict()
    correlations = numeric.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    report = {"source_profile": profile, "analysis_partition": "training only", "training_rows": len(train),
              "amount_by_class": amount_summary, "iqr_outlier_counts": {k: int(v) for k, v in outliers.items()},
              "target_correlations": correlations.to_dict(), "plots": plots,
              "findings": [f"The source fraud prevalence is {profile['fraud_percentage']:.4f}%; always predicting legitimate would look accurate.",
                           f"{profile['duplicate_rows']:,} exact duplicate rows are removed before splitting to avoid overlap.",
                           "No outlier rows are discarded: rare extremes can be useful fraud signals. Robust scaling and log amount reduce scale sensitivity.",
                           ("V1–V28 are anonymized PCA coordinates; their business meanings cannot be recovered." if "V28" in train else "Transaction category, amount and location signals are modeled. Names, card numbers and record identifiers are excluded."),
                           "Time is elapsed seconds from the configured source origin, not a verified local timezone.",
                           ("Chronological train/validation/test windows evaluate later transactions; timestamps tied at boundaries stay together. Customers can recur across windows, so this is not an unseen-customer benchmark." if profile.get("split_strategy") == "chronological" else "Random stratification is a benchmark. Temporal evaluation is needed before deployment on changing patterns.")]}
    (output / "eda.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "findings.md").write_text("# Dataset and EDA findings\n\n" + "\n\n".join(report["findings"]) +
                                        "\n\nAll distribution and correlation charts use only the training partition.\n", encoding="utf-8")
    return report
