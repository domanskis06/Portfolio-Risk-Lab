"""Orchestrate feature computation and persist to `features` table.

Pipeline:
  prices_clean → (per instrument) ret_1d, vol_21d
               → beta_63d vs benchmark returns
               → upsert features (instrument_id, dt, ret_1d, vol_21d, beta_63d)
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert

from src.db.connection import get_engine
from src.features.returns import simple_returns
from src.features.rolling import BETA_WINDOW, VOL_WINDOW, rolling_beta, rolling_vol

BENCHMARK_TICKER = "SPY"
_CHUNK_SIZE = 1000


def load_clean_prices() -> pd.DataFrame:
    """Return DataFrame: instrument_id, dt, adj_close (sorted)."""
    query = """
        SELECT instrument_id, dt, adj_close
        FROM prices_clean
        ORDER BY instrument_id, dt
    """
    return pd.read_sql(query, get_engine())


def load_benchmark_instrument_id(ticker: str = BENCHMARK_TICKER) -> int:
    """Return instruments.instrument_id for ticker (SPY)."""
    query = text(
        "SELECT instrument_id FROM instruments WHERE ticker = :ticker"
    )
    with get_engine().connect() as conn:
        instrument_id = conn.execute(query, {"ticker": ticker}).scalar_one()
    return int(instrument_id)


def compute_feature_frame(prices: pd.DataFrame, benchmark_id: int) -> pd.DataFrame:
    """Build in-memory features: instrument_id, dt, ret_1d, vol_21d, beta_63d."""
    prices = prices.sort_values(by=["instrument_id", "dt"]).copy()

    # Index by date so rolling_beta aligns asset and benchmark on dates
    bench = prices.loc[prices["instrument_id"] == benchmark_id].set_index("dt")["adj_close"]
    benchmark_ret = simple_returns(bench)

    frames: list[pd.DataFrame] = []
    for instrument_id in prices["instrument_id"].unique():
        inst = prices.loc[prices["instrument_id"] == instrument_id].set_index("dt")
        instrument_ret = simple_returns(inst["adj_close"])
        instrument_vol = rolling_vol(instrument_ret, VOL_WINDOW)
        instrument_beta = rolling_beta(instrument_ret, benchmark_ret, BETA_WINDOW)

        frames.append(
            pd.DataFrame(
                {
                    "instrument_id": instrument_id,
                    "dt": instrument_ret.index,
                    "ret_1d": instrument_ret.to_numpy(),
                    "vol_21d": instrument_vol.to_numpy(),
                    "beta_63d": instrument_beta.reindex(instrument_ret.index).to_numpy(),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def upsert_features(features: pd.DataFrame) -> int:
    """Upsert into features on PK (instrument_id, dt)."""
    if features.empty:
        return 0

    cols = ["instrument_id", "dt", "ret_1d", "vol_21d", "beta_63d"]
    records = features[cols].to_dict(orient="records")

    engine = get_engine()
    table = Table("features", MetaData(), autoload_with=engine)

    total = 0
    with engine.begin() as conn:
        for i in range(0, len(records), _CHUNK_SIZE):
            chunk = records[i : i + _CHUNK_SIZE]
            stmt = insert(table).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["instrument_id", "dt"],
                set_={
                    "ret_1d": stmt.excluded.ret_1d,
                    "vol_21d": stmt.excluded.vol_21d,
                    "beta_63d": stmt.excluded.beta_63d,
                },
            )
            result = conn.execute(stmt)
            total += result.rowcount or 0
    return total


def build_and_store_features() -> int:
    """Compute features for all instruments and upsert them; return affected rows."""
    prices = load_clean_prices()
    bid = load_benchmark_instrument_id()
    feat = compute_feature_frame(prices, bid)
    return upsert_features(feat)
