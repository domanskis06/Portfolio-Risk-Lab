"""Build the feature table from prices_clean.

Usage (ETL already run):
    python scripts/run_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.features.build_feature_table import build_and_store_features

    n = build_and_store_features()
    print(f"Upsert features: {n}")


if __name__ == "__main__":
    main()
