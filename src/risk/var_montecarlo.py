"""Backward-compatible alias for src.risk.montecarlo."""

from src.risk.montecarlo import (
    monte_carlo_es,
    monte_carlo_var,
    monte_carlo_var_es,
    simulate_multivariate_normal,
)

__all__ = [
    "simulate_multivariate_normal",
    "monte_carlo_var_es",
    "monte_carlo_var",
    "monte_carlo_es",
]
