"""Rolling volatility, correlation, beta vs benchmark.

Beta (definition):
    beta_hat = Cov(R, R_b) / Var(R_b)
i.e. how much the asset moves with the benchmark, in units of benchmark variance.

TODO (Step 4 — you implement).
"""

from __future__ import annotations

import pandas as pd


def rolling_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualization policy: document when implementing (daily std vs *sqrt(252))."""
    raise NotImplementedError("TODO Step 4: implement rolling_vol()")


def rolling_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling beta of asset vs benchmark."""
    raise NotImplementedError("TODO Step 4: implement rolling_beta()")
