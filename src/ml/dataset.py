"""Time-aware datasets and purged splits (no random shuffle on time series).

Label idea (Option A): high/low realized vol on horizon [t+1, t+h].
Features must use information only up to date t.

TODO (Step 8 — you implement).
"""

from __future__ import annotations

import pandas as pd


def make_vol_regime_labels(
    returns: pd.Series,
    horizon: int = 21,
    high_vol_quantile: float = 0.7,
) -> pd.Series:
    """Binary label: 1 = high forward realized vol, 0 = low."""
    raise NotImplementedError("TODO Step 8: implement make_vol_regime_labels()")


def purged_time_split(
    n_samples: int,
    n_splits: int = 5,
    purge_gap: int = 21,
) -> list[tuple]:
    """Return list of (train_idx, test_idx) with a purge gap between folds."""
    raise NotImplementedError("TODO Step 8: implement purged_time_split()")
