"""Minimal yfinance returns helper for quick experiments (not part of the ETL).

Ported from Market Risk Engine (market_risk.data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def load_returns_yfinance(
    tickers: list[str],
    start: str = "2021-01-01",
    end: str | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Download Adj Close, dropna, simple returns.
    Returns:
      returns: (T, N) float array
      index: dates of returns (length T)
      tickers: column order
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    prices = raw["Adj Close"].dropna(how="any")
    prices = prices[tickers]
    rets = prices.pct_change().dropna()
    return rets.to_numpy(), rets.index, list(prices.columns)
