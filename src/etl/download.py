"""Download OHLCV for a small liquid universe (yfinance or CSV).

TODO (Step 2 — you implement):
- define UNIVERSE tickers + benchmark
- download adjusted OHLCV
- return a tidy DataFrame: ticker, dt, open, high, low, close, adj_close, volume
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_ohlcv(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV for tickers.

    Args:
        tickers: Yahoo Finance symbols.
        start: Start date YYYY-MM-DD.
        end: Optional end date YYYY-MM-DD.

    Returns:
        Tidy frame with one row per (ticker, dt).
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    # raw ma MultiIndex kolumn: ('Open','SPY'), ('Adj Close','SPY'), ...
    # Zrób tidy: stack ticker albo pętla po tickerach.

    frames = []
    for t in tickers:
        sub = raw.xs(t, axis=1, level=1)
        sub = sub.reset_index()
        sub = sub.rename(columns={
            "Date": "dt",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        sub["ticker"] = t
        frames.append(sub[["ticker", "dt", "open", "high", "low", "close", "adj_close", "volume"]])
    
    out = pd.concat(frames, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"]).dt.date
    out = out.dropna(subset=["adj_close"])
    # yfinance leaves MultiIndex level name "Price" on columns — drop it
    out.columns.name = None
    return out