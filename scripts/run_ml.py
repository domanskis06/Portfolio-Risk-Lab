"""Train volatility-regime baseline + logreg and print metrics / run_ids.

Usage (venv, Docker up, features loaded):
    python scripts/run_ml.py
    python scripts/run_ml.py --ticker SPY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Vol regime ML walk-forward")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--horizon", type=int, default=21, help="Forward-vol label horizon")
    parser.add_argument("--quantile", type=float, default=0.7, help="High-vol quantile")
    parser.add_argument(
        "--model",
        choices=("logreg", "hgb"),
        default="logreg",
        help="logreg (stable) or hgb (can win on longer horizons)",
    )
    parser.add_argument("--no-persist", action="store_true", help="Skip DB writes")
    args = parser.parse_args()

    from src.ml.volatility_regime import train_vol_regime_model

    summary = train_vol_regime_model(
        ticker=args.ticker,
        horizon=args.horizon,
        high_vol_quantile=args.quantile,
        model_kind=args.model,
        persist=not args.no_persist,
    )
    print(
        f"ticker={summary['ticker']}  instrument_id={summary['instrument_id']}  "
        f"model={summary['model_kind']}  horizon={summary['horizon']}"
    )
    print(f"n_rows={summary['n_rows']}  folds={summary['n_folds_used']}")
    print(
        f"baseline  AUC={summary['baseline']['roc_auc']}  "
        f"AP={summary['baseline']['average_precision']}"
    )
    print(
        f"model     AUC={summary['model']['roc_auc']}  "
        f"AP={summary['model']['average_precision']}"
    )
    print(f"auc_lift  {summary['auc_lift']}")
    if summary["run_ids"]:
        print(f"run_ids={summary['run_ids']}")


if __name__ == "__main__":
    main()
