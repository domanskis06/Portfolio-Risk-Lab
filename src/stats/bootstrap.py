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
    block_size: int = 10,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high)."""


    rng = np.random.default_rng(seed)
    n = len(series)
    point = float(statistic(series))
    n_blocks = n // block_size

    if n_blocks < 1:
        raise ValueError("series shorter than block size")    
    
    usable = series[: n_blocks * block_size]
    blocks = usable.reshape(n_blocks, block_size)

    boot_stats = []

    for _ in range(n_boot):
        chosen = rng.integers(0, n_blocks, size=n_blocks) 
        sample = blocks[chosen].reshape(-1)              
        boot_stats.append(statistic(sample))

    ci_low, ci_high = np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)])
    return point, float(ci_low), float(ci_high)