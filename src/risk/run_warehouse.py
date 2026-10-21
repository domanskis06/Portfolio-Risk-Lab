"""Run H / P / MC risk on a DB portfolio and persist rows to risk_runs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from src.db.connection import get_engine
from src.portfolio.portfolio_returns import portfolio_losses
from src.risk.covariance import estimate_mean_cov, portfolio_mean_std
from src.risk.historical import historical_es, historical_var
from src.risk.montecarlo import monte_carlo_var_es
from src.risk.parametric import parametric_es_gaussian, parametric_var_gaussian

ALPHA = 0.95
HORIZON_DAYS = 1
MC_N_SIMS = 50_000
MC_SEED = 42


def load_portfolio_weights(portfolio_id: int) -> pd.Series:
    """Return Series indexed by instrument_id with portfolio weights."""
    engine = get_engine()
    query = text(
        """
        SELECT instrument_id, weight
        FROM portfolio_weights
        WHERE portfolio_id = :portfolio_id
        ORDER BY instrument_id
        """
    )
    with engine.connect() as conn:
        result = conn.execute(query, {"portfolio_id": portfolio_id})
        df = pd.DataFrame(result.fetchall(), columns=["instrument_id", "weight"])
    if df.empty:
        raise ValueError(f"No weights for portfolio_id={portfolio_id}")
    return df.set_index("instrument_id")["weight"].astype(float)


def load_returns_panel(
    instrument_ids: list[int] | pd.Index,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return DataFrame index=dt, columns=instrument_id, values=ret_1d (dropna)."""
    ids = [int(i) for i in instrument_ids]
    if not ids:
        raise ValueError("instrument_ids is empty")

    clauses = ["instrument_id IN :instrument_ids", "ret_1d IS NOT NULL"]
    params: dict[str, Any] = {"instrument_ids": ids}
    if start is not None:
        clauses.append("dt >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("dt <= :end")
        params["end"] = end

    sql = f"""
        SELECT dt, instrument_id, ret_1d
        FROM features
        WHERE {' AND '.join(clauses)}
        ORDER BY dt, instrument_id
    """
    query = text(sql).bindparams(bindparam("instrument_ids", expanding=True))

    with get_engine().connect() as conn:
        result = conn.execute(query, params)
        df = pd.DataFrame(result.fetchall(), columns=["dt", "instrument_id", "ret_1d"])

    if df.empty:
        raise ValueError("No feature returns for the requested instruments/dates")

    panel = (
        df.pivot(index="dt", columns="instrument_id", values="ret_1d")
        .sort_index()
        .dropna()
    )
    return panel


def run_risk_methods(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = ALPHA,
) -> dict[str, tuple[float, float]]:
    """Compute historical / parametric / MC VaR+ES. returns shape (T, N)."""
    losses = portfolio_losses(returns, weights)
    mu, cov = estimate_mean_cov(returns)
    mu_p, sigma_p = portfolio_mean_std(mu, cov, weights)

    var_h = historical_var(losses, alpha)
    es_h = historical_es(losses, alpha)

    var_p = parametric_var_gaussian(mu_p, sigma_p, alpha)
    es_p = parametric_es_gaussian(mu_p, sigma_p, alpha)

    var_mc, es_mc = monte_carlo_var_es(
        mu, cov, weights, alpha=alpha, n_sims=MC_N_SIMS, seed=MC_SEED
    )

    return {
        "historical": (var_h, es_h),
        "parametric": (var_p, es_p),
        "monte_carlo": (var_mc, es_mc),
    }


def insert_risk_run(
    portfolio_id: int,
    method: str,
    var_value: float,
    es_value: float,
    params: dict[str, Any],
    *,
    alpha: float = ALPHA,
    horizon_days: int = HORIZON_DAYS,
    window_start: Any = None,
    window_end: Any = None,
) -> int:
    """INSERT one risk_runs row; return run_id."""
    query = text(
        """
        INSERT INTO risk_runs (
            portfolio_id, method, alpha, horizon_days,
            window_start, window_end, var_value, es_value, params_json
        )
        VALUES (
            :portfolio_id, :method, :alpha, :horizon_days,
            :window_start, :window_end, :var_value, :es_value,
            CAST(:params_json AS jsonb)
        )
        RETURNING run_id
        """
    )
    with get_engine().begin() as conn:
        result = conn.execute(
            query,
            {
                "portfolio_id": portfolio_id,
                "method": method,
                "alpha": alpha,
                "horizon_days": horizon_days,
                "window_start": window_start,
                "window_end": window_end,
                "var_value": float(var_value),
                "es_value": float(es_value),
                "params_json": json.dumps(params),
            },
        )
        return int(result.scalar_one())


def run_and_store(portfolio_id: int) -> list[int]:
    """Load weights+returns from DB, run 3 methods, insert 3 risk_runs rows."""
    weights = load_portfolio_weights(portfolio_id)
    returns = load_returns_panel(weights.index)
    returns = returns.reindex(columns=weights.index).dropna()

    risk_results = run_risk_methods(
        returns.to_numpy(dtype=float),
        weights.to_numpy(dtype=float),
    )

    window_start = returns.index.min()
    window_end = returns.index.max()
    n_obs = int(len(returns))

    run_ids: list[int] = []
    for method, (var_value, es_value) in risk_results.items():
        params: dict[str, Any] = {"n_obs": n_obs, "n_assets": int(len(weights))}
        if method == "monte_carlo":
            params.update({"n_sims": MC_N_SIMS, "seed": MC_SEED})
        run_id = insert_risk_run(
            portfolio_id,
            method,
            var_value,
            es_value,
            params,
            window_start=window_start,
            window_end=window_end,
        )
        run_ids.append(run_id)
    return run_ids
