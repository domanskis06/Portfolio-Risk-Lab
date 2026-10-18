"""Expected Shortfall helpers.

``historical_es`` lives with historical VaR (engine layout); re-exported here
for the lab module path ``src.risk.expected_shortfall``.
"""

from __future__ import annotations

import numpy as np

from src.risk.var_historical import historical_es

__all__ = ["historical_es", "expected_shortfall"]


def expected_shortfall(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Alias for historical_es (lab stub name). alpha = confidence level."""
    return historical_es(losses, alpha=alpha)
