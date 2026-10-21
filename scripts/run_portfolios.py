"""Seed the equal weight and manual tilt portfolios into Postgres.

Usage (ETL already run):
    python scripts/run_portfolios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TICKERS = [
    "SPY",
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


def main() -> None:
    from src.portfolio.seed_portfolios import seed_default_portfolios

    ids = seed_default_portfolios(DEFAULT_TICKERS)
    print(f"Seeded portfolios: {ids}")


if __name__ == "__main__":
    main()
