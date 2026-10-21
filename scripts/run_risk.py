"""Run historical, parametric and Monte Carlo VaR/ES for portfolios and store them in risk_runs.

Usage (features and portfolios seeded):
    python scripts/run_risk.py                 # all portfolios
    python scripts/run_risk.py equal_universe  # one portfolio by name
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from sqlalchemy import bindparam, text

    from src.db.connection import get_engine
    from src.risk.run_warehouse import run_and_store

    names = sys.argv[1:]
    with get_engine().connect() as conn:
        if names:
            query = text(
                "SELECT portfolio_id, name FROM portfolios WHERE name IN :names ORDER BY portfolio_id"
            ).bindparams(bindparam("names", expanding=True))
            portfolios = conn.execute(query, {"names": names}).fetchall()
        else:
            portfolios = conn.execute(
                text("SELECT portfolio_id, name FROM portfolios ORDER BY portfolio_id")
            ).fetchall()
    if not portfolios:
        raise SystemExit("No matching portfolios, run scripts/run_portfolios.py first")

    result_query = text(
        "SELECT run_id, method, var_value, es_value FROM risk_runs WHERE run_id IN :ids ORDER BY run_id"
    ).bindparams(bindparam("ids", expanding=True))

    for portfolio_id, name in portfolios:
        run_ids = run_and_store(int(portfolio_id))
        print(f"{name} (portfolio_id={portfolio_id}) risk_runs: {run_ids}")
        with get_engine().connect() as conn:
            rows = conn.execute(result_query, {"ids": run_ids}).fetchall()
        for run_id, method, var_value, es_value in rows:
            print(f"  {run_id:>4}  {method:12}  VaR={var_value:.6f}  ES={es_value:.6f}")


if __name__ == "__main__":
    main()
