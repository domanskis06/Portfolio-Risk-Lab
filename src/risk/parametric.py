"""Gaussian parametric VaR and ES from portfolio mean and volatility.

Ported from Market Risk Engine (market_risk.parametric).
``alpha`` = confidence level (default 0.95), not tail probability 0.05.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.risk.covariance import portfolio_mean_std


def parametric_var_gaussian(mu_p: float, sigma_p: float, alpha: float = 0.95) -> float:
    """VaR_alpha = -mu_p + sigma_p * norm.ppf(alpha)."""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be in (0, 1)")
    if sigma_p < 0:
        raise ValueError("Sigma_p must be non-negative")
    return float(-mu_p + sigma_p * norm.ppf(alpha))


def parametric_es_gaussian(mu_p: float, sigma_p: float, alpha: float = 0.95) -> float:
    """ES_alpha = -mu_p + sigma_p * norm.pdf(norm.ppf(alpha)) / (1 - alpha)."""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be in (0, 1)")
    if sigma_p < 0:
        raise ValueError("Sigma_p must be non-negative")
    return float(-mu_p + sigma_p * norm.pdf(norm.ppf(alpha)) / (1 - alpha))


def _mu_vector(mean: float | np.ndarray, n: int) -> np.ndarray:
    if np.isscalar(mean):
        return np.full(n, float(mean))
    return np.asarray(mean, dtype=float)


def parametric_var(
    weights: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.95,
    mean: float | np.ndarray = 0.0,
) -> float:
    """Build (mu_p, sigma_p) from weights/cov, then Gaussian VaR."""
    mu = _mu_vector(mean, cov.shape[0])
    mu_p, sigma_p = portfolio_mean_std(mu, cov, weights)
    return parametric_var_gaussian(mu_p, sigma_p, alpha=alpha)


def parametric_es(
    weights: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.95,
    mean: float | np.ndarray = 0.0,
) -> float:
    """Build (mu_p, sigma_p) from weights/cov, then Gaussian ES."""
    mu = _mu_vector(mean, cov.shape[0])
    mu_p, sigma_p = portfolio_mean_std(mu, cov, weights)
    return parametric_es_gaussian(mu_p, sigma_p, alpha=alpha)
