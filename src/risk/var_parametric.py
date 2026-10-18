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


def parametric_var(
    weights: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.95,
    mean: float | np.ndarray = 0.0,
) -> float:
    """Thin lab wrapper: build (mu_p, sigma_p) then parametric_var_gaussian.

    If ``mean`` is a scalar, use constant mean vector of that value (length N).
    If ``mean`` is an array, use it as asset mean vector mu.
    """
    n = cov.shape[0]
    if np.isscalar(mean):
        mu = np.full(n, float(mean))
    else:
        mu = np.asarray(mean, dtype=float)
    mu_p, sigma_p = portfolio_mean_std(mu, cov, weights)
    return parametric_var_gaussian(mu_p, sigma_p, alpha=alpha)
