"""Load downloaded frames into prices_raw (idempotent upsert).

TODO (Step 2 — you implement):
- upsert on PRIMARY KEY (ticker, dt)
- set ingested_at
"""

from __future__ import annotations

import pandas as pd


def upsert_prices_raw(df: pd.DataFrame) -> int:
    """Upsert rows into prices_raw.

    Returns:
        Number of rows written / affected (define convention in docstring when implementing).
    """
    raise NotImplementedError("TODO Step 2: implement upsert_prices_raw()")
