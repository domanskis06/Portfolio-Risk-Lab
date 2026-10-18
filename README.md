# Quant Research & Market Risk Warehouse

Repeatable, point-in-time market-data pipeline with PostgreSQL storage, portfolio risk (historical / parametric / Monte Carlo), bootstrap uncertainty, and one ML module (volatility regime classification). Streamlit dashboard reads from the database.

## Conventions (MVP)

- Daily **simple** returns: \(R_t = P_t / P_{t-1} - 1\)
- Loss for VaR/ES: \(L = -R_p\)
- **`alpha` = confidence level** (default **0.95** for 95% VaR), not tail probability 0.05
- Historical VaR = `np.quantile(losses, alpha)`
- Long-only weights summing to 1
- Fixed `RANDOM_SEED` for MC / bootstrap / ML
- Timestamps in DB stored in UTC
- Every experiment stored as a `*_runs` row with `params_json`

## Risk engine port

Core risk math (historical / parametric / Monte Carlo VaR & ES, Kupiec backtest, stress)
is ported from the finished [Market-Risk-Engine](https://github.com/domanskis06/Market-Risk-Engine)
into `src/risk/` and `src/portfolio/`. Warehouse ETL / DB / ML stubs are separate and may still be unfinished.

```powershell
python -m pytest -q
python scripts/run_risk.py
```

## Stack

Python 3.12 · PostgreSQL 16 (Docker) · pandas / numpy / scipy / scikit-learn · SQLAlchemy · Streamlit · pytest

## Setup (Windows)

Prerequisites: Python 3.12, Docker Desktop running.

```powershell
cd C:\Users\szymo\Desktop\portfolio-risk-lab
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
docker compose up -d
python -m pytest -q
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Stop the database: `docker compose down` (data volume kept). Wipe volume: `docker compose down -v`.

## Project layout

See `src/`, `sql/`, `scripts/`, `tests/`. Implementation proceeds step-by-step (Mode A: stubs first, then user fills logic).

## ML module (chosen)

**Option A — volatility regime classification** (high / low realized vol on a short forward horizon), purged time-aware split, AUC / precision-recall.

## Limitations (honest, filled as we go)

- Small liquid universe; not survivorship-free
- yfinance quality / corporate actions are imperfect
- Gaussian / i.i.d. assumptions in parametric VaR and naive bootstrap are simplifications
