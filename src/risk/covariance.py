"""Mean/covariance and portfolio variance helpers.

Core estimators ported from Market Risk Engine (market_risk.parametric):
estimate_mean_cov, portfolio_mean_std.
Convenience wrappers: sample_covariance, portfolio_variance.
"""

from __future__ import annotations

import numpy as np

from src.portfolio.weights import validate_weights


def estimate_mean_cov(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """returns (T, N) → (mu (N,), cov (N, N))."""
    mu = np.mean(returns, axis=0)
    cov = np.cov(returns, rowvar=False)
    return mu, cov


def portfolio_mean_std(
    mu: np.ndarray, cov: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Return (mu_p, sigma_p)."""
    validate_weights(weights)
    mu_p = float(np.dot(weights, mu))
    sigma_p = float(np.sqrt(weights @ cov @ weights))
    return mu_p, sigma_p


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Thin wrapper: cov part of estimate_mean_cov. returns shape (T, N)."""
    _, cov = estimate_mean_cov(returns)
    return cov


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    """Return w^T Sigma w as a float."""
    validate_weights(weights)
    return float(weights @ cov @ weights)
