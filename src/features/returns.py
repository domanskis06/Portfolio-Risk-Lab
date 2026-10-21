"""Simple returns from adjusted close prices."""

from __future__ import annotations

import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns R_t = P_t / P_{t-1} - 1; the first value is NaN."""
    return prices.pct_change()
