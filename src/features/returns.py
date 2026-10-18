"""Simple returns from adjusted close.

Simple return (definition):
    R_t = P_t / P_{t-1} - 1
i.e. today's price over yesterday's price, minus one.

TODO (Step 4 — you implement).
"""

from __future__ import annotations

import pandas as pd


def simple_returns(prices: pd.Series) -> pd.Series:
    """Compute daily simple returns from a price series indexed by date."""
    raise NotImplementedError("TODO Step 4: implement simple_returns()")
