"""Simple returns from adjusted close.

Definition (say it out loud):
    R_t = P_t / P_{t-1} - 1
i.e. today's price divided by yesterday's price, minus one.

Example: P_{t-1}=100, P_t=102 → R_t = 0.02 (plus two percent that day).
"""

from __future__ import annotations

import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns; index aligned with prices (first day = NaN).

    TODO: one-liner is fine — pct_change() OR prices / prices.shift(1) - 1
    """
    raise NotImplementedError("TODO Step 4: implement simple_returns()")
