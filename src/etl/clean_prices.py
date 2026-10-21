"""Transform prices_raw into prices_clean.

prices_raw keeps whatever Yahoo returned. prices_clean holds one adjusted close
per (instrument_id, dt), sorted, with a quality_flag. Tickers are mapped to a
stable integer instrument_id because symbols can be renamed or reused.

Missing values: rows without an adjusted close are dropped, nothing is forward filled.
Outliers: days with an absolute simple return above RETURN_CAP are flagged
``outlier_return`` but kept.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import MetaData, Table, select
from sqlalchemy.dialects.postgresql import insert

from src.db.connection import get_engine

RETURN_CAP = 0.25
BENCHMARK_TICKER = "SPY"
_CHUNK_SIZE = 1000


def ensure_instruments(tickers: list[str], benchmark: str = BENCHMARK_TICKER) -> pd.DataFrame:
    """Upsert tickers into instruments; return DataFrame[ticker, instrument_id]."""
    engine = get_engine()
    metadata = MetaData()
    table = Table("instruments", metadata, autoload_with=engine)

    with engine.begin() as conn:
        for t in tickers:
            stmt = insert(table).values(ticker=t, is_benchmark=(t == benchmark))
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker"],
                set_={"is_benchmark": stmt.excluded.is_benchmark},
            )
            conn.execute(stmt)

        rows = conn.execute(
            select(table.c.ticker, table.c.instrument_id).where(table.c.ticker.in_(tickers))
        ).fetchall()

    return pd.DataFrame(rows, columns=["ticker", "instrument_id"])


def load_prices_raw(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load from prices_raw into a DataFrame ordered by ticker, dt."""
    engine = get_engine()
    metadata = MetaData()
    table = Table("prices_raw", metadata, autoload_with=engine)

    with engine.connect() as conn:
        stmt = select(table).order_by(table.c.ticker, table.c.dt)
        if tickers is not None:
            stmt = stmt.where(table.c.ticker.in_(tickers))
        result = conn.execute(stmt)
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def clean_price_panel(raw: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Clean raw panel → instrument_id, dt, adj_close, quality_flag."""
    df = raw.merge(id_map, on="ticker", how="left")
    df = df.sort_values(by=["instrument_id", "dt"])
    df = df.dropna(subset=["adj_close"])

    # pct_change per instrument, otherwise the first day of one ticker uses the last price of another
    df["return"] = df.groupby("instrument_id", sort=False)["adj_close"].pct_change()

    df["quality_flag"] = "ok"
    outlier = df["return"].abs() > RETURN_CAP
    df.loc[outlier.fillna(False), "quality_flag"] = "outlier_return"

    return df[["instrument_id", "dt", "adj_close", "quality_flag"]].reset_index(drop=True)


def upsert_prices_clean(clean: pd.DataFrame) -> int:
    """Upsert into prices_clean on PRIMARY KEY (instrument_id, dt)."""
    if clean.empty:
        return 0

    cols = ["instrument_id", "dt", "adj_close", "quality_flag"]
    records = clean[cols].to_dict(orient="records")

    engine = get_engine()
    metadata = MetaData()
    table = Table("prices_clean", metadata, autoload_with=engine)

    total = 0
    with engine.begin() as conn:
        for i in range(0, len(records), _CHUNK_SIZE):
            chunk = records[i : i + _CHUNK_SIZE]
            stmt = insert(table).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["instrument_id", "dt"],
                set_={
                    "adj_close": stmt.excluded.adj_close,
                    "quality_flag": stmt.excluded.quality_flag,
                },
            )
            result = conn.execute(stmt)
            total += result.rowcount or 0
    return total


def run_cleaning(tickers: list[str] | None = None) -> int:
    """Orchestrate: ensure instruments → load raw → clean → upsert clean."""
    if not tickers:
        raise ValueError("tickers list is required for run_cleaning()")
    id_map = ensure_instruments(tickers)
    raw = load_prices_raw(tickers)
    clean = clean_price_panel(raw, id_map)
    return upsert_prices_clean(clean)
