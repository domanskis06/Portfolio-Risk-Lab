"""Non-overlapping block bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    series: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    block_size: int = 10,
) -> tuple[float, float, float]:
    """Percentile CI for ``statistic`` using a non-overlapping block bootstrap.

    The series is cut into consecutive blocks of ``block_size`` observations.
    Blocks are resampled with replacement and concatenated, which keeps short
    range dependence such as volatility clustering inside each block.
    Trailing observations that do not fill a block are dropped.

    Returns:
        (point_estimate, ci_low, ci_high) for a ``1 - alpha`` interval.
    """
    rng = np.random.default_rng(seed)
    n_blocks = len(series) // block_size
    if n_blocks < 1:
        raise ValueError("series shorter than block size")

    point = float(statistic(series))
    blocks = series[: n_blocks * block_size].reshape(n_blocks, block_size)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        boot_stats[i] = statistic(blocks[chosen].reshape(-1))

    ci_low, ci_high = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(ci_low), float(ci_high)
