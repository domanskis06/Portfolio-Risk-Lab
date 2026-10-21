"""Monte Carlo VaR and ES under a multivariate normal return model.

Ported from Market Risk Engine (market_risk.monte_carlo).
``alpha`` = confidence level (default 0.95).
"""

from __future__ import annotations

import numpy as np

from src.portfolio.portfolio_returns import portfolio_losses
from src.portfolio.weights import validate_weights
from src.risk.historical import historical_es, historical_var


def simulate_multivariate_normal(
    mu: np.ndarray,
    cov: np.ndarray,
    n_sims: int,
    seed: int | None = None,
) -> np.ndarray:
    """Return array shape (n_sims, N)."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n_sims)


def monte_carlo_var_es(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sims: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Simulate → portfolio losses → (historical_var, historical_es) on those losses."""
    validate_weights(weights)
    losses = portfolio_losses(
        simulate_multivariate_normal(mu, cov, n_sims, seed), weights
    )
    var_mc = historical_var(losses, alpha)
    es_mc = historical_es(losses, alpha)
    return var_mc, es_mc


def monte_carlo_var(
    weights: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.95,
    n_sims: int = 10_000,
    seed: int = 42,
) -> float:
    """Return only the VaR from monte_carlo_var_es."""
    var_mc, _ = monte_carlo_var_es(
        mean, cov, weights, alpha=alpha, n_sims=n_sims, seed=seed
    )
    return var_mc

def monte_carlo_es(
    weights: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.95,
    n_sims: int = 10_000,
    seed: int = 42,
) -> float:
    """Return only the ES from monte_carlo_var_es."""
    _, es_mc = monte_carlo_var_es(mean, cov, weights, alpha=alpha, n_sims=n_sims, seed=seed)
    return es_mc
