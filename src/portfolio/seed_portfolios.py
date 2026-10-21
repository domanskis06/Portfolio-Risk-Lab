"""Seed the default portfolios and their weights into Postgres."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import bindparam, text

from src.db.connection import get_engine
from src.portfolio.weights import equal_weights, validate_weights

VALID_FROM = "2019-01-01"

# Explicit tilt over the default universe, sums to 1.
_MANUAL_TILT = {
    "SPY": 0.15,
    "QQQ": 0.20,
    "IWM": 0.10,
    "EFA": 0.10,
    "EEM": 0.05,
    "TLT": 0.05,
    "GLD": 0.10,
    "XLE": 0.10,
    "XLF": 0.10,
    "XLV": 0.05,
}


def load_ticker_ids(tickers: list[str]) -> pd.DataFrame:
    """Return DataFrame[ticker, instrument_id] for the given tickers."""
    if not tickers:
        return pd.DataFrame(columns=["ticker", "instrument_id"])

    engine = get_engine()
    query = text(
        "SELECT ticker, instrument_id FROM instruments WHERE ticker IN :tickers"
    ).bindparams(bindparam("tickers", expanding=True))

    with engine.connect() as conn:
        result = conn.execute(query, {"tickers": list(tickers)})
        return pd.DataFrame(result.fetchall(), columns=["ticker", "instrument_id"])


def create_or_get_portfolio(name: str, description: str) -> int:
    """Insert portfolio or update description on name conflict; return portfolio_id."""
    engine = get_engine()
    query = text(
        """
        INSERT INTO portfolios (name, description)
        VALUES (:name, :description)
        ON CONFLICT (name) DO UPDATE
          SET description = EXCLUDED.description
        RETURNING portfolio_id
        """
    )
    with engine.begin() as conn:
        result = conn.execute(query, {"name": name, "description": description})
        return int(result.scalar_one())


def upsert_weights(
    portfolio_id: int,
    weights: pd.Series,
    ticker_ids: pd.DataFrame,
    valid_from: str = VALID_FROM,
) -> int:
    """Upsert weight rows into portfolio_weights. valid_to stays NULL (open-ended)."""
    if weights.empty:
        return 0

    id_map = ticker_ids.set_index("ticker")["instrument_id"]
    missing = [t for t in weights.index if t not in id_map.index]
    if missing:
        raise KeyError(f"Tickers not in instruments / ticker_ids: {missing}")

    engine = get_engine()
    query = text(
        """
        INSERT INTO portfolio_weights
          (portfolio_id, instrument_id, weight, valid_from, valid_to)
        VALUES
          (:portfolio_id, :instrument_id, :weight, :valid_from, :valid_to)
        ON CONFLICT (portfolio_id, instrument_id, valid_from) DO UPDATE
          SET weight = EXCLUDED.weight
        """
    )

    total = 0
    with engine.begin() as conn:
        for ticker, weight in weights.items():
            result = conn.execute(
                query,
                {
                    "portfolio_id": portfolio_id,
                    "instrument_id": int(id_map.loc[ticker]),
                    "weight": float(weight),
                    "valid_from": valid_from,
                    "valid_to": None,
                },
            )
            total += result.rowcount or 0
    return total


def _manual_tilt_weights(tickers: list[str]) -> pd.Series:
    """Build manual tilt Series for the requested tickers (renormalize if subset)."""
    w = pd.Series({t: _MANUAL_TILT[t] for t in tickers if t in _MANUAL_TILT}, dtype=float)
    if w.empty:
        raise ValueError("No overlapping tickers with _MANUAL_TILT")
    w = w / w.sum()
    validate_weights(w.to_numpy())
    return w


def seed_default_portfolios(tickers: list[str]) -> dict[str, int]:
    """Create equal_universe + manual_tilt portfolios and upsert their weights."""
    ticker_ids = load_ticker_ids(tickers)
    if ticker_ids.empty:
        raise ValueError("No instruments found — run ETL/cleaning first")

    pid = create_or_get_portfolio(
        "equal_universe",
        "Equal weight across the research universe",
    )
    upsert_weights(pid, equal_weights(tickers), ticker_ids)

    pid2 = create_or_get_portfolio(
        "manual_tilt",
        "Manual tilt: overweight QQQ, underweight TLT and EEM",
    )
    upsert_weights(pid2, _manual_tilt_weights(tickers), ticker_ids)

    return {"equal_universe": pid, "manual_tilt": pid2}
