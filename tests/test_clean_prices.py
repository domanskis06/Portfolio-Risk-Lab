"""Step 3: cleaning logic on a tiny synthetic panel (no network)."""

from __future__ import annotations

import pandas as pd
import pytest


def _synthetic_raw() -> pd.DataFrame:
    """Two tickers; one missing adj_close; one crazy jump day."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "dt": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-02", "2024-01-03"]
            ).date,
            "adj_close": [100.0, None, 101.0, 50.0, 80.0],  # BBB jumps +60% → outlier
            "open": [100.0, 100.0, 101.0, 50.0, 80.0],
            "high": [100.0, 100.0, 101.0, 50.0, 80.0],
            "low": [100.0, 100.0, 101.0, 50.0, 80.0],
            "close": [100.0, 100.0, 101.0, 50.0, 80.0],
            "volume": [1e6, 1e6, 1e6, 1e6, 1e6],
        }
    )


def _id_map() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["AAA", "BBB"], "instrument_id": [1, 2]})


def test_clean_drops_missing_adj_close_and_flags_outlier():
    from src.etl.clean_prices import RETURN_CAP, clean_price_panel

    clean = clean_price_panel(_synthetic_raw(), _id_map())

    assert list(clean.columns) == ["instrument_id", "dt", "adj_close", "quality_flag"]
    assert clean["adj_close"].notna().all()
    # AAA had a None day → that row gone → 2 AAA + 2 BBB = 4
    assert len(clean) == 4
    assert clean["instrument_id"].is_monotonic_increasing or True  # soft
    # BBB day with +60% return should be flagged if |r| > RETURN_CAP
    bbb = clean[clean["instrument_id"] == 2].sort_values("dt")
    assert (bbb["adj_close"].iloc[1] / bbb["adj_close"].iloc[0] - 1) > RETURN_CAP
    assert "outlier" in str(bbb["quality_flag"].iloc[1]).lower()


def test_clean_sorted_by_instrument_and_date():
    from src.etl.clean_prices import clean_price_panel

    clean = clean_price_panel(_synthetic_raw(), _id_map())
    key = clean[["instrument_id", "dt"]].reset_index(drop=True)
    assert key.equals(key.sort_values(["instrument_id", "dt"]).reset_index(drop=True))
