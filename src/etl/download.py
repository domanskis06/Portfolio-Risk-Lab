"""Download OHLCV for a small liquid universe (yfinance or CSV).

TODO (Step 2 — you implement):
- define UNIVERSE tickers + benchmark
- download adjusted OHLCV
- return a tidy DataFrame: ticker, dt, open, high, low, close, adj_close, volume
"""

from __future__ import annotations

import pandas as pd


def download_ohlcv(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV for tickers.

    Args:
        tickers: Yahoo Finance symbols.
        start: Start date YYYY-MM-DD.
        end: Optional end date YYYY-MM-DD.

    Returns:
        Tidy frame with one row per (ticker, dt).
    """
    raise NotImplementedError("TODO Step 2: implement download_ohlcv()")
