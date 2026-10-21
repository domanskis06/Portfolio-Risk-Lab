"""Expected Shortfall helpers.

``historical_es`` lives with historical VaR and is re-exported here.
"""

from __future__ import annotations

import numpy as np

from src.risk.var_historical import historical_es

__all__ = ["historical_es", "expected_shortfall"]


def expected_shortfall(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Alias for historical_es. alpha is the confidence level."""
    return historical_es(losses, alpha=alpha)
