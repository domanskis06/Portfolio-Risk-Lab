"""Step 4 unit tests — pure math on tiny series (no DB / no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_simple_returns_manual():
    from src.features.returns import simple_returns

    prices = pd.Series([100.0, 102.0, 101.0], index=pd.RangeIndex(3))
    r = simple_returns(prices)
    assert np.isnan(r.iloc[0])
    assert np.isclose(r.iloc[1], 0.02)
    assert np.isclose(r.iloc[2], 101.0 / 102.0 - 1.0)


def test_rolling_vol_constant_series_near_zero():
    from src.features.rolling import rolling_vol

    # constant returns → std ≈ 0 after window fills
    r = pd.Series([0.01] * 40)
    vol = rolling_vol(r, window=21, annualize=False)
    assert np.isclose(vol.iloc[-1], 0.0, atol=1e-12)


def test_rolling_beta_known_case():
    from src.features.rolling import rolling_beta

    # asset = 2 * benchmark → beta should be ≈ 2 on a long enough window
    rng = np.random.default_rng(0)
    rb = pd.Series(rng.normal(0, 0.01, size=100))
    ra = 2.0 * rb
    beta = rolling_beta(ra, rb, window=50)
    assert np.isclose(beta.iloc[-1], 2.0, rtol=1e-6)
