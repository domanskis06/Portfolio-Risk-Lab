"""Portfolio returns and losses from asset returns and weights.

Ported from Market Risk Engine (market_risk.portfolio).
Convention: L = -R_p.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.weights import validate_weights


def portfolio_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """returns: (T, N), weights: (N,). Return R_p shape (T,)."""
    validate_weights(weights)
    return np.dot(returns, weights)


def portfolio_losses(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """L = -R_p."""
    validate_weights(weights)
    return -portfolio_returns(returns, weights)


def portfolio_return(weights: np.ndarray, returns_row: np.ndarray) -> float:
    """Thin wrapper: one-day portfolio return (lab stub name)."""
    return float(portfolio_returns(np.atleast_2d(returns_row), weights)[0])


def portfolio_return_series(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """weights indexed by ticker; returns columns = tickers, index = dates."""
    cols = list(weights.index)
    w = validate_weights(weights.loc[cols].to_numpy(dtype=float))
    rp = portfolio_returns(returns[cols].to_numpy(dtype=float), w)
    return pd.Series(rp, index=returns.index, name="portfolio_return")
