"""Time-aware datasets and purged splits (no random shuffle on time series).

Label (Option A): high/low *forward* realized vol on horizon [t+1, t+h].
Features used elsewhere must use information only up to date t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_vol_regime_labels(
    returns: pd.Series,
    horizon: int = 21,
    high_vol_quantile: float = 0.7,
) -> pd.Series:
    """Binary label: 1 = high forward realized vol, 0 = low; NaN if no future window.

    Forward vol at date t is std(r_{t+1}, ..., r_{t+h}).
    Threshold = ``high_vol_quantile`` of the non-null forward-vol series.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 < high_vol_quantile < 1.0:
        raise ValueError("high_vol_quantile must be in (0, 1)")

    r = returns.astype(float)
    # Reverse rolling → at t: std(r[t], ..., r[t+h-1]); shift(-1) → std(r[t+1], ..., r[t+h])
    forward_vol = r.iloc[::-1].rolling(horizon).std().iloc[::-1].shift(-1)
    valid = forward_vol.notna()
    if not valid.any():
        return pd.Series(np.nan, index=r.index, dtype=float)

    thresh = float(forward_vol.loc[valid].quantile(high_vol_quantile))
    labels = pd.Series(np.nan, index=r.index, dtype=float)
    labels.loc[valid] = (forward_vol.loc[valid] >= thresh).astype(float)
    return labels


def purged_time_split(
    n_samples: int,
    n_splits: int = 5,
    purge_gap: int = 21,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds with a purge gap between train end and test start.

    Fold k::

        [======== train ========][ gap ][== test ==]
        0 ..................... train_end
                                 test_start ...... test_end

    ``purge_gap`` should be >= label horizon so train labels do not overlap test features.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if purge_gap < 0:
        raise ValueError("purge_gap must be >= 0")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    fold_size = n_samples // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(
            f"n_samples={n_samples} too small for n_splits={n_splits}"
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits else n_samples
        train_end = test_start - purge_gap
        if train_end <= 0 or test_end <= test_start:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError(
            f"No valid folds for n_samples={n_samples}, "
            f"n_splits={n_splits}, purge_gap={purge_gap}"
        )
    return splits
