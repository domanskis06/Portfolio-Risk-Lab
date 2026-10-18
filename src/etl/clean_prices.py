"""Transform prices_raw → prices_clean with quality flags.

TODO (Step 3 — you implement):
- sort by date, use adj_close
- missing-value policy (document: ffill vs leave-missing)
- basic outlier caps
- write quality_flag
"""

from __future__ import annotations

import pandas as pd


def clean_price_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw OHLCV panel into instrument_id, dt, adj_close, quality_flag."""
    raise NotImplementedError("TODO Step 3: implement clean_price_panel()")
