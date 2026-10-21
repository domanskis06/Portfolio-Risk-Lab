"""Rolling volatility and beta vs a benchmark.

Volatility is the trailing standard deviation of daily returns, annualized with
sqrt(252) by default.

Beta is Cov(R, R_b) / Var(R_b) computed inside a trailing window.

Both use pandas rolling windows that end at the current row, so the value at
date t only depends on returns dated t or earlier.
"""

from __future__ import annotations

from math import sqrt

import pandas as pd

VOL_WINDOW = 21
BETA_WINDOW = 63


def rolling_vol(returns: pd.Series, window: int = VOL_WINDOW, annualize: bool = True) -> pd.Series:
    """Trailing std of returns; multiplied by sqrt(252) when ``annualize`` is set."""
    s = returns.rolling(window).std()
    return s * sqrt(252) if annualize else s


def rolling_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = BETA_WINDOW,
) -> pd.Series:
    """Trailing beta of the asset against the benchmark on their common dates."""
    a, b = asset_returns.align(benchmark_returns, join="inner")
    return a.rolling(window).cov(b) / b.rolling(window).var()
