"""Download daily OHLCV for a list of tickers from Yahoo Finance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

OUTPUT_COLUMNS = ["ticker", "dt", "open", "high", "low", "close", "adj_close", "volume"]


def download_ohlcv(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV for tickers.

    Args:
        tickers: Yahoo Finance symbols.
        start: Start date YYYY-MM-DD.
        end: Optional end date YYYY-MM-DD.

    Returns:
        Tidy frame with one row per (ticker, dt). Rows without an adjusted close
        are dropped.
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    rename = {
        "Date": "dt",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    frames = []
    for ticker in tickers:
        sub = raw.xs(ticker, axis=1, level=1).reset_index().rename(columns=rename)
        sub["ticker"] = ticker
        frames.append(sub[OUTPUT_COLUMNS])

    out = pd.concat(frames, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"]).dt.date
    out = out.dropna(subset=["adj_close"])
    out.columns.name = None
    return out
