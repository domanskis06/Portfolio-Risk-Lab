"""Rolling VaR forecasts and Kupiec proportion-of-failures backtest.

Ported from Market Risk Engine (market_risk.backtest).
``alpha`` = confidence level (default 0.95).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from src.risk.var_historical import historical_var


def exception_indicator(losses: np.ndarray, var: float | np.ndarray) -> np.ndarray:
    """1 where loss > var; var may be scalar or same-length array."""
    return (losses > var).astype(int)


def kupiec_pof_test(
    losses: np.ndarray,
    var: float | np.ndarray,
    alpha: float = 0.95,
) -> dict:
    """
    Keys: n, x, hit_ratio, expected_p, lr_stat, p_value, reject_5pct
    Safe log when hat_pi is 0 or 1 (clip with tiny eps).
    """
    hit_ratio = np.mean(exception_indicator(losses, var))
    expected_p = 1 - alpha
    n = len(losses)
    x = int(np.sum(exception_indicator(losses, var)))
    hat_pi = np.clip(x / n, 1e-12, 1 - 1e-12)

    loglik_p = (n - x) * np.log(1 - expected_p) + x * np.log(expected_p)
    loglik_hat = (n - x) * np.log(1 - hat_pi) + x * np.log(hat_pi)
    lr = -2.0 * (loglik_p - loglik_hat)
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    reject_5pct = bool(p_value <= 0.05)
    return {
        "n": n,
        "x": x,
        "hit_ratio": float(hit_ratio),
        "expected_p": expected_p,
        "lr_stat": float(lr),
        "p_value": p_value,
        "reject_5pct": reject_5pct,
    }


def rolling_historical_var(
    losses: np.ndarray,
    window: int,
    alpha: float = 0.95,
) -> np.ndarray:
    """
    var[t] = historical_var(losses[t-window:t], alpha) for t >= window
    var[:window] = nan
    """
    var = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        var[t] = historical_var(losses[t - window : t], alpha)
    return var


def rolling_kupiec(
    losses: np.ndarray,
    window: int,
    alpha: float = 0.95,
) -> dict:
    """Align losses[window:] with var[window:]; call kupiec_pof_test."""
    var = rolling_historical_var(losses, window, alpha)
    return kupiec_pof_test(losses[window:], var[window:], alpha)
