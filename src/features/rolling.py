"""Rolling volatility and beta vs benchmark.

Volatility (intuition):
  How wildly returns jump day-to-day. MVP: rolling standard deviation of returns.
  Annualization (document your choice in the docstring when you implement):
    daily_std * sqrt(252)  ≈ annualized vol (common market convention)
  Schema column is vol_21d — either store daily or annualized; be consistent.

Beta (definition):
    beta_hat = Cov(R, R_b) / Var(R_b)
  i.e. how many units the asset moves when the benchmark moves one unit of variance.
  Rolling: compute cov/var inside a trailing window (e.g. 63 trading days).

Point-in-time:
  Window ending at date t may only use returns with dates <= t (pandas rolling does this).
"""

from __future__ import annotations

import pandas as pd

# defaults matching sql/001_schema.sql column names
VOL_WINDOW = 21
BETA_WINDOW = 63


def rolling_vol(returns: pd.Series, window: int = VOL_WINDOW, annualize: bool = True) -> pd.Series:
    """Trailing std of returns; if annualize, multiply by sqrt(252).

    TODO:
      s = returns.rolling(window).std()
      return s * sqrt(252) if annualize else s
    """
    raise NotImplementedError("TODO Step 4: implement rolling_vol()")


def rolling_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = BETA_WINDOW,
) -> pd.Series:
    """Trailing beta of asset vs benchmark; align indexes first.

    TODO sketch:
      a, b = asset_returns.align(benchmark_returns, join="inner")
      cov = a.rolling(window).cov(b)
      var = b.rolling(window).var()
      return cov / var
    """
    raise NotImplementedError("TODO Step 4: implement rolling_beta()")
