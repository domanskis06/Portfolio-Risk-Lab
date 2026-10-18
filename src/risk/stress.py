"""Simple stress / what-if scenarios.

Ported from Market Risk Engine (market_risk.stress).
"""

from __future__ import annotations

import numpy as np

from src.portfolio.weights import validate_weights


def apply_parallel_shock(weights: np.ndarray, shock: float = -0.10) -> float:
    """
    One-day scenario: every asset return equals `shock`.
    For long-only weights summing to 1, portfolio return = shock.
    Returns portfolio LOSS = -shock (e.g. shock=-0.10 -> loss=+0.10).
    """
    validate_weights(weights)
    portfolio_return = float(np.dot(weights, np.full(len(weights), shock)))
    return -portfolio_return


def scale_cov_vol(cov: np.ndarray, vol_mult: float = 2.0) -> np.ndarray:
    """
    Multiply all volatilities by vol_mult.
    Because cov entries scale with sigma_i * sigma_j, cov *= vol_mult**2.
    """
    if vol_mult <= 0:
        raise ValueError("vol_mult must be positive")
    return cov * (vol_mult**2)


def stressed_portfolio_loss_parallel(weights: np.ndarray, shock: float = -0.10) -> float:
    """Wrapper: parallel-shock portfolio loss."""
    return apply_parallel_shock(weights, shock=shock)
