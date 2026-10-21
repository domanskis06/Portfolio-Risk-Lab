"""Streamlit dashboard — reads all analytics from PostgreSQL.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` without editable install
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.risk.covariance import estimate_mean_cov, portfolio_mean_std
from src.risk.historical import historical_es, historical_var
from src.risk.montecarlo import monte_carlo_var_es
from src.risk.parametric import parametric_es_gaussian, parametric_var_gaussian
from src.stats.bootstrap import bootstrap_ci
from src.viz import plots
from src.viz import queries as q

st.set_page_config(
    page_title="Portfolio Risk Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }
    h1 {
        font-weight: 650 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem !important;
    }
    section[data-testid="stSidebar"] {
        background: #0A101C;
        border-right: 1px solid #1F2937;
    }
    button[data-baseweb="tab"] {
        font-size: 1rem;
        padding: 0.6rem 1.1rem;
    }
    .soft-caption {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin: 0 0 1rem 0;
    }
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.4rem 0 1.1rem 0;
    }
    .kpi-card {
        background: #151D2E;
        border: 1px solid #243044;
        border-radius: 12px;
        padding: 0.9rem 1.05rem;
        min-width: 0;
    }
    .kpi-label {
        color: #9CA3AF;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 0.35rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        color: #E8EDF5;
        font-size: 1.55rem;
        font-weight: 650;
        letter-spacing: -0.02em;
        line-height: 1.2;
        word-break: break-word;
    }
    @media (max-width: 900px) {
        .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def _portfolios() -> pd.DataFrame:
    return q.list_portfolios()


@st.cache_data(ttl=60, show_spinner=False)
def _date_bounds():
    return q.feature_date_bounds()


@st.cache_data(ttl=60, show_spinner=False)
def _panel(portfolio_id: int, start: str, end: str):
    returns, weights = q.load_portfolio_panel(portfolio_id, start=start, end=end)
    perf = q.portfolio_nav_drawdown(returns, weights)
    return returns, weights, perf


@st.cache_data(ttl=60, show_spinner=False)
def _risk_runs(portfolio_id: int) -> pd.DataFrame:
    return q.load_latest_risk_runs(portfolio_id)


@st.cache_data(ttl=120, show_spinner="Computing live VaR/ES…")
def _live_risk(portfolio_id: int, start: str, end: str) -> pd.DataFrame:
    """Compute H/P/MC VaR+ES on the selected window when DB has no risk_runs.

    Uses a lighter Monte Carlo (10k paths) so the dashboard stays responsive.
    """
    returns, weights = q.load_portfolio_panel(portfolio_id, start=start, end=end)
    R = returns.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    losses = -(R @ w)
    mu, cov = estimate_mean_cov(R)
    mu_p, sigma_p = portfolio_mean_std(mu, cov, w)
    alpha = 0.95

    var_h, es_h = historical_var(losses, alpha), historical_es(losses, alpha)
    var_p = parametric_var_gaussian(mu_p, sigma_p, alpha)
    es_p = parametric_es_gaussian(mu_p, sigma_p, alpha)
    var_mc, es_mc = monte_carlo_var_es(
        mu, cov, w, alpha=alpha, n_sims=10_000, seed=42
    )

    results = {
        "historical": (var_h, es_h),
        "parametric": (var_p, es_p),
        "monte_carlo": (var_mc, es_mc),
    }
    rows = [
        {
            "method": method,
            "alpha": alpha,
            "var_value": var_v,
            "es_value": es_v,
            "window_start": pd.Timestamp(returns.index.min()).date(),
            "window_end": pd.Timestamp(returns.index.max()).date(),
        }
        for method, (var_v, es_v) in results.items()
    ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner=False)
def _model_runs() -> pd.DataFrame:
    return q.list_model_runs(limit=30)


@st.cache_data(ttl=60, show_spinner=False)
def _predictions(run_id: int) -> pd.DataFrame:
    return q.load_predictions(run_id)


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{digits}%}"


def _kpi_row(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def _bootstrap_hist_var(losses: np.ndarray, alpha: float = 0.95):
    """Block-bootstrap CI for historical VaR on portfolio losses."""
    if len(losses) < 40:
        return None
    point, lo, hi = bootstrap_ci(
        losses.astype(float),
        statistic=lambda s: historical_var(np.asarray(s, dtype=float), alpha=alpha),
        n_boot=500,
        alpha=0.05,
        seed=42,
        block_size=10,
    )
    return point, lo, hi


def _chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Controls")

try:
    portfolios = _portfolios()
    dmin, dmax = _date_bounds()
except Exception as exc:  # noqa: BLE001 — show DB errors in UI
    st.error(
        "Cannot reach PostgreSQL. Start Docker (`docker compose up -d`) "
        "and check `DATABASE_URL` in `.env`."
    )
    st.caption(str(exc))
    st.stop()

if portfolios.empty or dmin is None or dmax is None:
    st.warning("Database is empty — run ETL, features, seed portfolios, risk, and ML first.")
    st.stop()

name_to_id = dict(zip(portfolios["name"], portfolios["portfolio_id"], strict=True))
portfolio_name = st.sidebar.selectbox("Portfolio", list(name_to_id.keys()))
portfolio_id = int(name_to_id[portfolio_name])

date_range = st.sidebar.date_input(
    "Date range",
    value=(dmin.date(), dmax.date()),
    min_value=dmin.date(),
    max_value=dmax.date(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = dmin.date(), dmax.date()

if st.sidebar.button("Refresh cache", width="stretch"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("All charts read from Postgres — no in-memory globals.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
desc = portfolios.loc[portfolios["portfolio_id"] == portfolio_id, "description"]
desc_txt = "" if desc.empty or pd.isna(desc.iloc[0]) else str(desc.iloc[0])

st.title("Portfolio Risk Lab")
st.markdown(
    f'<p class="soft-caption">{portfolio_name}'
    + (f" — {desc_txt}" if desc_txt else "")
    + f" · {start_date} → {end_date}</p>",
    unsafe_allow_html=True,
)

try:
    returns, weights, perf = _panel(portfolio_id, str(start_date), str(end_date))
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

total_ret = float(perf["nav"].iloc[-1] - 1.0)
max_dd = float(perf["drawdown"].min())
vol_ann = float(perf["ret"].std() * np.sqrt(252))
n_days = int(len(perf))

_kpi_row(
    [
        ("Total return", _fmt_pct(total_ret)),
        ("Max drawdown", _fmt_pct(max_dd)),
        ("Ann. volatility", _fmt_pct(vol_ann)),
        ("Trading days", f"{n_days:,}"),
    ]
)

# ---------------------------------------------------------------------------
# Sections (tabs; heavy results are cached with st.cache_data)
# ---------------------------------------------------------------------------
tab_perf, tab_risk, tab_ml = st.tabs(["Performance", "Risk", "ML regime"])

with tab_perf:
    _chart(plots.nav_drawdown_figure(perf))
    _chart(plots.weights_bar_figure(weights))

with tab_risk:
    stored = _risk_runs(portfolio_id)
    if stored.empty:
        risk_df = _live_risk(portfolio_id, str(start_date), str(end_date))
        source_note = "computed live for selected window"
        table_caption = (
            "No stored `risk_runs` for this portfolio — showing live VaR/ES "
            "on the selected date range. Persist with `python scripts/run_risk.py`."
        )
    else:
        risk_df = stored
        source_note = "from risk_runs (latest per method)"
        table_caption = "Latest stored experiment per method from `risk_runs`."

    losses = (-perf["ret"]).to_numpy(dtype=float)
    boot = _bootstrap_hist_var(losses, alpha=0.95)

    _chart(plots.correlation_heatmap(returns))

    chart_col, table_col = st.columns([1.55, 1], gap="large")
    with chart_col:
        if boot is None:
            _chart(plots.var_comparison_figure(risk_df, source_note=source_note))
        else:
            point, lo, hi = boot
            _chart(
                plots.var_comparison_figure(
                    risk_df,
                    bootstrap_point=point,
                    bootstrap_low=lo,
                    bootstrap_high=hi,
                    source_note=source_note,
                )
            )
            st.caption(
                f"Amber diamond: block-bootstrap 95% CI for historical VaR "
                f"on the selected window — {_fmt_pct(point, 3)} "
                f"[{_fmt_pct(lo, 3)}, {_fmt_pct(hi, 3)}]."
            )
    with table_col:
        st.markdown("##### Risk figures")
        st.caption(table_caption)
        show = risk_df[["method", "var_value", "es_value"]].copy()
        if "alpha" in risk_df.columns:
            show.insert(1, "alpha", risk_df["alpha"])
        show["var_value"] = show["var_value"].map(lambda x: f"{x:.3%}")
        show["es_value"] = show["es_value"].map(lambda x: f"{x:.3%}")
        show["method"] = show["method"].astype(str).str.replace("_", " ")
        st.dataframe(show, width="stretch", hide_index=True)

with tab_ml:
    runs = _model_runs()
    if runs.empty:
        st.info(
            "No `model_runs` yet. Train with "
            "`python -c \"from src.ml.volatility_regime import train_vol_regime_model; "
            "print(train_vol_regime_model())\"`."
        )
    else:
        # Real models first; the baseline is only a reference run.
        is_base = runs["model_name"].eq("vol_regime_baseline")
        runs = pd.concat([runs[~is_base], runs[is_base]], ignore_index=True)
        labels = []
        for _, row in runs.iterrows():
            metrics = q.parse_json_col(row["metrics_json"])
            auc = metrics.get("roc_auc")
            auc_s = f"AUC {auc:.3f}" if isinstance(auc, (int, float)) else "AUC —"
            tag = " (reference)" if row["model_name"] == "vol_regime_baseline" else ""
            labels.append(f"#{row['run_id']} · {row['model_name']}{tag} · {auc_s}")

        choice = st.selectbox("Model run", labels, index=0)
        idx = labels.index(choice)
        run_id = int(runs.iloc[idx]["run_id"])
        row = runs.iloc[idx]
        metrics = q.parse_json_col(row["metrics_json"])
        params = q.parse_json_col(row["params_json"])

        baseline_auc = None
        for _, r in runs.iterrows():
            if r["model_name"] != "vol_regime_baseline":
                continue
            p = q.parse_json_col(r["params_json"])
            if p.get("ticker") == params.get("ticker") and p.get("horizon") == params.get(
                "horizon"
            ):
                baseline_auc = q.parse_json_col(r["metrics_json"]).get("roc_auc")
                break

        auc_v = metrics.get("roc_auc")
        ap_v = metrics.get("average_precision")
        lift = (
            f"{auc_v - baseline_auc:+.3f}"
            if isinstance(auc_v, (int, float)) and isinstance(baseline_auc, (int, float))
            else "—"
        )
        _kpi_row(
            [
                ("ROC AUC", f"{auc_v:.3f}" if isinstance(auc_v, (int, float)) else "—"),
                (
                    "Avg precision",
                    f"{ap_v:.3f}" if isinstance(ap_v, (int, float)) else "—",
                ),
                ("AUC lift vs baseline", lift),
                (
                    "Ticker / horizon",
                    f"{params.get('ticker', '—')} · h={params.get('horizon', '—')}",
                ),
            ]
        )

        pred = _predictions(run_id)
        _chart(plots.ml_regime_figure(pred))
        st.caption(
            "Top: out-of-sample probability of a high-vol regime. "
            "Bottom: trailing 21-day realized vol. "
            "Rose bands = periods that were truly high-vol. "
            "The baseline score is the percentile of current volatility in the training window."
        )

        with st.expander("Run params / metrics JSON"):
            st.json({"metrics": metrics, "params": params})

