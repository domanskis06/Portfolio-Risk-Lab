"""Step 2 tests: tidy OHLCV shape + idempotent upsert into prices_raw.

These fail with NotImplementedError until you implement download/staging.
After implementation, run with Docker up:
    python -m pytest tests/test_etl_staging.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.db.connection import get_engine


REQUIRED_COLS = {
    "ticker",
    "dt",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
}


def _fake_ohlcv() -> pd.DataFrame:
    """Tiny synthetic panel — use this in upsert tests (no network)."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "dt": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]).date,
            "open": [1.0, 1.1, 2.0],
            "high": [1.2, 1.2, 2.1],
            "low": [0.9, 1.0, 1.9],
            "close": [1.1, 1.15, 2.05],
            "adj_close": [1.1, 1.15, 2.05],
            "volume": [1000.0, 1100.0, 2000.0],
        }
    )


def test_download_ohlcv_tidy_columns():
    from src.etl.download import download_ohlcv

    df = download_ohlcv(["SPY"], start="2024-01-02", end="2024-01-10")
    assert REQUIRED_COLS.issubset(set(df.columns))
    assert df["ticker"].nunique() >= 1
    assert not df.duplicated(subset=["ticker", "dt"]).any()


def test_upsert_prices_raw_idempotent():
    from src.etl.staging import upsert_prices_raw

    df = _fake_ohlcv()
    n1 = upsert_prices_raw(df)
    n2 = upsert_prices_raw(df)  # second run must not create duplicate PK rows

    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM prices_raw WHERE ticker IN ('AAA', 'BBB')")
        ).scalar_one()

    assert n1 >= 1
    assert count == 3  # still exactly 3 unique (ticker, dt) rows
    # n2 can be 0 or 3 depending on how you count "affected"; just no crash / no dupes
    assert n2 >= 0
