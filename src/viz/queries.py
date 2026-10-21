"""DB loaders for the Streamlit dashboard — all reads go through Postgres."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine
from src.risk.run_warehouse import load_portfolio_weights, load_returns_panel


def list_portfolios() -> pd.DataFrame:
    """Return portfolio_id, name, description ordered by id."""
    q = text(
        """
        SELECT portfolio_id, name, description
        FROM portfolios
        ORDER BY portfolio_id
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn)


def feature_date_bounds() -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Min/max dates available in features."""
    q = text("SELECT MIN(dt) AS dmin, MAX(dt) AS dmax FROM features")
    with get_engine().connect() as conn:
        row = conn.execute(q).mappings().one()
    if row["dmin"] is None:
        return None, None
    return pd.Timestamp(row["dmin"]), pd.Timestamp(row["dmax"])


def load_weights_with_tickers(portfolio_id: int) -> pd.DataFrame:
    """Weights joined to tickers for a portfolio."""
    q = text(
        """
        SELECT pw.instrument_id, i.ticker, pw.weight
        FROM portfolio_weights pw
        JOIN instruments i ON i.instrument_id = pw.instrument_id
        WHERE pw.portfolio_id = :portfolio_id
        ORDER BY i.ticker
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"portfolio_id": int(portfolio_id)})
    if df.empty:
        raise ValueError(f"No weights for portfolio_id={portfolio_id}")
    return df


def load_portfolio_panel(
    portfolio_id: int,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (returns panel with ticker columns, weight Series indexed by ticker)."""
    meta = load_weights_with_tickers(portfolio_id)
    id_to_ticker = dict(zip(meta["instrument_id"], meta["ticker"], strict=True))
    weights_by_id = load_portfolio_weights(portfolio_id)

    start_s = None if start is None else str(pd.Timestamp(start).date())
    end_s = None if end is None else str(pd.Timestamp(end).date())
    panel = load_returns_panel(weights_by_id.index, start=start_s, end=end_s)
    panel = panel.reindex(columns=weights_by_id.index).dropna()
    panel = panel.rename(columns=id_to_ticker)
    panel.columns.name = None
    panel.index.name = None

    weights = (
        meta.set_index("ticker")["weight"]
        .astype(float)
        .reindex(panel.columns)
    )
    weights.name = "weight"
    return panel, weights


def portfolio_nav_drawdown(
    returns: pd.DataFrame, weights: pd.Series
) -> pd.DataFrame:
    """Build daily portfolio return, NAV (start=1), and drawdown series."""
    cols = list(weights.index)
    rp = returns[cols].to_numpy(dtype=float) @ weights.loc[cols].to_numpy(dtype=float)
    out = pd.DataFrame({"ret": rp}, index=pd.DatetimeIndex(returns.index))
    out["nav"] = (1.0 + out["ret"]).cumprod()
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1.0
    return out


def load_latest_risk_runs(portfolio_id: int) -> pd.DataFrame:
    """Latest risk_runs row per method for a portfolio."""
    q = text(
        """
        SELECT DISTINCT ON (method)
            run_id, portfolio_id, method, alpha, horizon_days,
            window_start, window_end, var_value, es_value,
            params_json, created_at
        FROM risk_runs
        WHERE portfolio_id = :portfolio_id
        ORDER BY method, created_at DESC, run_id DESC
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params={"portfolio_id": int(portfolio_id)})


def list_model_runs(limit: int = 20) -> pd.DataFrame:
    """Recent ML model_runs (newest first)."""
    q = text(
        """
        SELECT run_id, model_name, train_start, train_end,
               test_start, test_end, metrics_json, params_json, created_at
        FROM model_runs
        ORDER BY created_at DESC, run_id DESC
        LIMIT :limit
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params={"limit": int(limit)})


def load_predictions(run_id: int) -> pd.DataFrame:
    """OOS predictions for one model run, joined with ticker + trailing vol."""
    q = text(
        """
        SELECT p.dt, p.instrument_id, i.ticker,
               p.y_true, p.y_pred, f.vol_21d
        FROM predictions p
        JOIN instruments i ON i.instrument_id = p.instrument_id
        LEFT JOIN features f
          ON f.instrument_id = p.instrument_id AND f.dt = p.dt
        WHERE p.run_id = :run_id
        ORDER BY p.dt
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"run_id": int(run_id)})
    if not df.empty:
        df["dt"] = pd.to_datetime(df["dt"])
    return df


def parse_json_col(value: Any) -> dict:
    """Normalize JSONB / str / dict into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        return json.loads(value)
    return dict(value)
