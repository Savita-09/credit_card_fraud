"""Measured fraud-class metrics, validation-only threshold selection and chart data."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def evaluate(y, probabilities, threshold):
    prediction = np.asarray(probabilities) >= threshold
    return {"accuracy": float(accuracy_score(y, prediction)), "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)), "f1": float(f1_score(y, prediction, zero_division=0)),
            "f2": float(fbeta_score(y, prediction, beta=2, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)), "pr_auc": float(average_precision_score(y, probabilities)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def optimize_threshold(y, probabilities, minimum_recall=None):
    if minimum_recall is not None and not 0 <= minimum_recall <= 1:
        raise ValueError("minimum_recall must be between 0 and 1.")
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-15)
    if minimum_recall is not None:
        f2 = np.where(recall[:-1] >= minimum_recall, f2, -np.inf)
    # Conservative tie-break: highest threshold among equally scoring validation candidates.
    candidates = np.flatnonzero(np.isclose(f2, f2.max(), atol=1e-12, rtol=0))
    return float(thresholds[candidates[-1]])


def curves(y, probabilities):
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    def thin(a, b, names):
        indices = np.unique(np.linspace(0, len(a) - 1, min(180, len(a))).astype(int))
        return [{names[0]: float(a[i]), names[1]: float(b[i])} for i in indices]
    return {"roc": thin(fpr, tpr, ["fpr", "tpr"]), "precision_recall": thin(recall[::-1], precision[::-1], ["recall", "precision"])}


def histogram(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    return [{"label": f"{edges[i]:.2g}–{edges[i+1]:.2g}", "count": int(count),
             "lower": float(edges[i]), "upper": float(edges[i+1])} for i, count in enumerate(counts)]
