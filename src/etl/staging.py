"""Load downloaded frames into prices_raw (idempotent upsert)."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import MetaData, Table, func
from sqlalchemy.dialects.postgresql import insert

from src.db.connection import get_engine

# Postgres has a practical limit on bind parameters; chunk large frames.
_CHUNK_SIZE = 1000


def upsert_prices_raw(df: pd.DataFrame) -> int:
    """Upsert rows into prices_raw on PRIMARY KEY (ticker, dt).

    Returns:
        Sum of rowcounts reported by the driver across chunks
        (inserts + updates). Re-running the same data is safe (idempotent).
    """
    if df.empty:
        return 0

    cols = ["ticker", "dt", "open", "high", "low", "close", "adj_close", "volume"]
    records = df[cols].to_dict(orient="records")

    engine = get_engine()
    metadata = MetaData()
    table = Table("prices_raw", metadata, autoload_with=engine)

    total = 0
    with engine.begin() as conn:
        for i in range(0, len(records), _CHUNK_SIZE):
            chunk = records[i : i + _CHUNK_SIZE]
            stmt = insert(table).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "dt"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "adj_close": stmt.excluded.adj_close,
                    "volume": stmt.excluded.volume,
                    "ingested_at": func.now(),
                },
            )
            result = conn.execute(stmt)
            total += result.rowcount or 0

    return total
