"""Step 8 unit tests — labels + purged splits (no DB)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_forward_labels_use_future_not_past():
    from src.ml.dataset import make_vol_regime_labels

    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.001, size=40)
    noisy = rng.normal(0, 0.05, size=40)
    r = pd.Series(np.concatenate([quiet, noisy]))
    h = 10

    labels = make_vol_regime_labels(r, horizon=h, high_vol_quantile=0.7)
    # Last h days have no full forward window → NaN
    assert labels.iloc[-h:].isna().all()

    # Manual forward vol at t=0 equals std(r[1:1+h])
    fwd0 = float(r.iloc[1 : 1 + h].std(ddof=1))
    assert np.isclose(
        r.iloc[::-1].rolling(h).std().iloc[::-1].shift(-1).iloc[0],
        fwd0,
    )

    # Day just before the noisy block: forward window is fully inside noisy → high
    assert labels.iloc[39] == 1.0


def test_purged_split_gap_and_order():
    from src.ml.dataset import purged_time_split

    splits = purged_time_split(n_samples=300, n_splits=4, purge_gap=21)
    assert len(splits) >= 1
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() - train_idx.max() - 1 >= 21
        assert train_idx.min() == 0  # expanding window


def test_classification_metrics_single_class():
    from src.ml.evaluate import classification_metrics

    m = classification_metrics(np.zeros(10), np.linspace(0, 1, 10))
    assert m["roc_auc"] is None
    assert m["n"] == 10
