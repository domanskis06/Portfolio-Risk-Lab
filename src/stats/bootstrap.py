"""Bootstrap confidence intervals (block or i.i.d. with README disclaimer).

TODO (Step 7 — you implement).
Target: CI for VaR or for portfolio Sharpe/mean — pick one and do it well.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    series: np.ndarray,
    statistic,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high)."""
    raise NotImplementedError("TODO Step 7: implement bootstrap_ci()")
