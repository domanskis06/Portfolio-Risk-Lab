"""Weight helpers. validate_weights ported from Market Risk Engine.

equal_weights remains a warehouse stub (not in the engine source).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_weights(weights: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Ensure 1-D weights, non-negative, sum to 1. Return float array.

    Raise ValueError with a clear message otherwise.
    """
    if weights.ndim != 1:
        raise ValueError("Weights must be a 1-D array")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    if np.abs(np.sum(weights) - 1) > tol:
        raise ValueError("Weights must sum to 1")
    return weights.astype(float)


def equal_weights(tickers: list[str]) -> pd.Series:
    """Return Series indexed by ticker with equal weights summing to 1."""
    return pd.Series(1/len(tickers), index = tickers)