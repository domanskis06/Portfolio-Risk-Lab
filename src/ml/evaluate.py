"""ML metrics: AUC, average precision; safe when a fold has one class only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return dict with roc_auc, average_precision (None if undefined), n, positive_rate.

    ``y_score`` should be P(class=1) or a ranking score — not hard 0/1 labels
    if you care about a meaningful AUC.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    out: dict = {
        "roc_auc": None,
        "average_precision": None,
        "n": int(y_true.size),
        "positive_rate": float(np.mean(y_true)) if y_true.size else None,
    }
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["average_precision"] = float(average_precision_score(y_true, y_score))
    return out
