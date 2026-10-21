"""Download prices from Yahoo Finance into prices_raw and build prices_clean.

Usage (database up, schema applied):
    python scripts/run_etl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TICKERS = [
    "SPY",  # benchmark
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "GLD",
    "XLE",
    "XLF",
    "XLV",
]
START = "2019-01-01"


def main() -> None:
    from src.etl.clean_prices import run_cleaning
    from src.etl.download import download_ohlcv
    from src.etl.staging import upsert_prices_raw

    print(f"Downloading {len(DEFAULT_TICKERS)} tickers from {START} ...")
    df = download_ohlcv(DEFAULT_TICKERS, start=START)
    print(f"Downloaded rows: {len(df)}")
    n = upsert_prices_raw(df)
    print(f"Upsert prices_raw: {n}")

    n_clean = run_cleaning(DEFAULT_TICKERS)
    print(f"Upsert prices_clean: {n_clean}")


if __name__ == "__main__":
    main()
