"""Run the whole pipeline end to end: schema, ETL, features, portfolios, risk, ML.

Usage (database up):
    python scripts/run_all_demo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ["init_db.py"],
    ["run_etl.py"],
    ["run_features.py"],
    ["run_portfolios.py"],
    ["run_risk.py"],
    ["run_ml.py", "--model", "logreg"],
]


def main() -> None:
    for step in STEPS:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / step[0]), *step[1:]], check=True)


if __name__ == "__main__":
    main()
