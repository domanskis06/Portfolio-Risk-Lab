"""Orchestrate feature computation and persist to `features` table.

Pipeline:
  prices_clean → (per instrument) ret_1d, vol_21d
               → beta_63d vs benchmark returns
               → upsert features (instrument_id, dt, ret_1d, vol_21d, beta_63d)
"""

from __future__ import annotations

import pandas as pd

from src.features.rolling import BETA_WINDOW, VOL_WINDOW

BENCHMARK_TICKER = "SPY"


def load_clean_prices() -> pd.DataFrame:
    """SELECT instrument_id, dt, adj_close FROM prices_clean ORDER BY ...

    TODO: use get_engine + pandas.read_sql or sqlalchemy select.
    """
    raise NotImplementedError("TODO Step 4: implement load_clean_prices()")


def load_benchmark_instrument_id(ticker: str = BENCHMARK_TICKER) -> int:
    """Lookup instruments.instrument_id for the benchmark ticker."""
    raise NotImplementedError("TODO Step 4: implement load_benchmark_instrument_id()")


def compute_feature_frame(prices: pd.DataFrame, benchmark_id: int) -> pd.DataFrame:
    """Build features table in memory.

    Args:
        prices: columns instrument_id, dt, adj_close (from prices_clean)
        benchmark_id: instrument_id of SPY (or chosen benchmark)

    Returns:
        columns: instrument_id, dt, ret_1d, vol_21d, beta_63d

    TODO steps:
      1. sort by instrument_id, dt
      2. for each instrument: simple_returns(adj_close) → ret_1d
      3. for each instrument: rolling_vol(ret_1d, VOL_WINDOW) → vol_21d
      4. take benchmark ret series; for each other (and SPY itself) rolling_beta
      5. concat into one DataFrame; NaNs at warm-up of rolling windows are OK
    """
    raise NotImplementedError("TODO Step 4: implement compute_feature_frame()")


def upsert_features(features: pd.DataFrame) -> int:
    """Upsert into features on PK (instrument_id, dt). Return rowcount sum."""
    raise NotImplementedError("TODO Step 4: implement upsert_features()")


def build_and_store_features() -> int:
    """Glue: load → compute → upsert. Return rows written."""
    raise NotImplementedError("TODO Step 4: implement build_and_store_features()")
