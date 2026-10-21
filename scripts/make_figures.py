"""Render the README figures from the warehouse into docs/img.

Usage (pipeline already run):
    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression

from src.ml import volatility_regime as vr
from src.risk.backtest import kupiec_pof_test, rolling_historical_var
from src.risk.covariance import estimate_mean_cov, portfolio_mean_std
from src.risk.historical import historical_var
from src.risk.parametric import parametric_var_gaussian
from src.stats.bootstrap import bootstrap_ci
from src.viz import queries as q

OUT = ROOT / "docs" / "img"

BG = "#0d1117"
PANEL = "#0d1117"
GRID = "#21262d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
AMBER = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "font.size": 10.5,
            "legend.frameon": False,
            "legend.labelcolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def performance(panels: dict[str, tuple[pd.DataFrame, pd.Series]]) -> dict:
    perf = {name: q.portfolio_nav_drawdown(*p) for name, p in panels.items()}
    equal_panel = panels["equal_universe"][0]
    spy = (1 + equal_panel["SPY"]).cumprod()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08}
    )
    ax1.plot(perf["equal_universe"].index, perf["equal_universe"]["nav"], color=BLUE, lw=1.8, label="equal_universe")
    ax1.plot(perf["manual_tilt"].index, perf["manual_tilt"]["nav"], color=GREEN, lw=1.5, label="manual_tilt")
    ax1.plot(spy.index, spy.values, color=MUTED, lw=1.2, ls="--", label="SPY buy and hold")
    ax1.set_ylabel("NAV (start = 1)")
    ax1.set_title("Portfolio NAV and drawdown")
    ax1.legend(loc="upper left", ncol=3)
    for name, color in (("equal_universe", BLUE), ("manual_tilt", GREEN)):
        ax2.fill_between(perf[name].index, perf[name]["drawdown"], 0, color=color, alpha=0.28, lw=0)
        ax2.plot(perf[name].index, perf[name]["drawdown"], color=color, lw=1)
    ax2.set_ylabel("Drawdown")
    pct_axis(ax2)
    save(fig, "performance.png")

    stats = {}
    for name, p in perf.items():
        n = len(p)
        years = n / 252
        stats[name] = {
            "total_return": float(p["nav"].iloc[-1] - 1),
            "cagr": float(p["nav"].iloc[-1] ** (1 / years) - 1),
            "ann_vol": float(p["ret"].std() * np.sqrt(252)),
            "sharpe_rf0": float(p["ret"].mean() / p["ret"].std() * np.sqrt(252)),
            "max_dd": float(p["drawdown"].min()),
            "days": n,
        }
    spy_ret = equal_panel["SPY"]
    spy_dd = spy / spy.cummax() - 1
    stats["SPY"] = {
        "total_return": float(spy.iloc[-1] - 1),
        "cagr": float(spy.iloc[-1] ** (252 / len(spy)) - 1),
        "ann_vol": float(spy_ret.std() * np.sqrt(252)),
        "sharpe_rf0": float(spy_ret.mean() / spy_ret.std() * np.sqrt(252)),
        "max_dd": float(spy_dd.min()),
        "days": len(spy),
    }
    return stats


def correlation(returns: pd.DataFrame) -> None:
    corr = returns.corr()
    cmap = LinearSegmentedColormap.from_list("rg", [BLUE, BG, RED])
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=9)
    ax.set_title("Daily return correlation")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    save(fig, "correlation.png")


def var_es(risk: pd.DataFrame, boot: tuple[float, float, float]) -> None:
    order = ["historical", "parametric", "monte_carlo"]
    risk = risk.set_index("method").loc[order]
    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    b1 = ax.bar(x - w / 2, risk["var_value"], w, color=BLUE, label="VaR 95%")
    b2 = ax.bar(x + w / 2, risk["es_value"], w, color=GREEN, label="ES 95%")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0004, f"{b.get_height():.2%}",
                    ha="center", va="bottom", color=TEXT, fontsize=9.5)
    point, lo, hi = boot
    ax.errorbar([x[0] - w / 2], [point], yerr=[[point - lo], [hi - point]], fmt="D", color=AMBER,
                capsize=5, lw=2, ms=6, label="Bootstrap 95% CI, historical VaR")
    ax.set_xticks(x, [o.replace("_", " ") for o in order])
    ax.set_ylim(0, risk["es_value"].max() * 1.5)
    pct_axis(ax, decimals=1)
    ax.set_ylabel("One day loss")
    ax.set_title("One day VaR and ES by method, equal_universe")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "var_es.png")


def bootstrap_distribution(losses: np.ndarray) -> tuple[float, float, float]:
    samples: list[float] = []

    def stat(s: np.ndarray) -> float:
        v = historical_var(np.asarray(s, dtype=float), 0.95)
        samples.append(v)
        return v

    point, lo, hi = bootstrap_ci(losses, stat, n_boot=2000, alpha=0.05, seed=42, block_size=10)
    boot = np.array(samples[1:])
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.hist(boot, bins=48, color=BLUE, alpha=0.85, edgecolor=BG)
    ax.axvline(point, color=AMBER, lw=2, label=f"Point estimate {point:.3%}")
    ax.axvline(lo, color=RED, lw=1.5, ls="--", label=f"2.5% / 97.5%: {lo:.3%} / {hi:.3%}")
    ax.axvline(hi, color=RED, lw=1.5, ls="--")
    pct_axis(ax, "x", 2)
    ax.set_xlabel("Historical VaR 95% on a bootstrap resample")
    ax.set_ylabel("Resamples")
    ax.set_title("Block bootstrap distribution of historical VaR (2000 resamples, block 10)")
    ax.legend()
    save(fig, "bootstrap_var.png")
    return point, lo, hi


def backtest(losses: pd.Series, window: int = 250) -> dict:
    arr = losses.to_numpy()
    var = rolling_historical_var(arr, window, 0.95)
    k = kupiec_pof_test(arr[window:], var[window:], 0.95)
    exc = arr > var
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(losses.index, arr, color=MUTED, lw=0.7, alpha=0.9, label="Daily portfolio loss")
    ax.plot(losses.index, var, color=BLUE, lw=1.6, label=f"Rolling {window} day historical VaR 95%")
    ax.scatter(losses.index[exc], arr[exc], s=14, color=RED, zorder=3, label=f"Exceptions ({k['x']})")
    ax.axhline(0, color=GRID, lw=1)
    pct_axis(ax)
    ax.set_ylabel("Loss (negative = gain)")
    ax.set_title("Rolling VaR backtest, equal_universe")
    ax.legend(loc="upper left", ncol=3)
    ax.text(
        0.995, 0.03,
        f"hit ratio {k['hit_ratio']:.2%} vs 5.00% expected\nKupiec p-value {k['p_value']:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", color=TEXT, fontsize=10,
    )
    save(fig, "backtest.png")
    return k


def ml_figures() -> dict:
    summary = vr.train_vol_regime_model(persist=False)
    folds = pd.DataFrame(summary["folds"])

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(folds))
    w = 0.38
    ax.bar(x - w / 2, folds["baseline_auc"], w, color=MUTED, label="Volatility persistence baseline")
    ax.bar(x + w / 2, folds["model_auc"], w, color=BLUE, label="Logistic regression")
    ax.axhline(0.5, color=RED, lw=1.2, ls="--", label="Random (0.5)")
    for i, r in folds.iterrows():
        ax.text(i - w / 2, r["baseline_auc"] + 0.01, f"{r['baseline_auc']:.2f}", ha="center", color=TEXT, fontsize=9)
        ax.text(i + w / 2, r["model_auc"] + 0.01, f"{r['model_auc']:.2f}", ha="center", color=TEXT, fontsize=9)
    ax.set_xticks(x, [f"Fold {int(f)}\n{s[:7]}\nto {e[:7]}" for f, s, e in zip(folds["fold"], folds["test_start"], folds["test_end"])])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC on test fold")
    ax.set_title("Out of sample AUC per walk forward fold, SPY, 21 day horizon")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "ml_folds.png")

    runs = q.list_model_runs()
    model_id = int(runs.loc[runs["model_name"] == "vol_regime_logreg", "run_id"].iloc[0])
    base_id = int(runs.loc[runs["model_name"] == "vol_regime_baseline", "run_id"].iloc[0])
    pred = q.load_predictions(model_id)
    base = q.load_predictions(base_id)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1}
    )
    flag = (pred["y_true"] >= 0.5).to_numpy()
    start = None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if (not f or i == len(flag) - 1) and start is not None:
            end = i if f else i - 1
            for ax in (ax1, ax2):
                ax.axvspan(pred["dt"].iloc[start] - pd.Timedelta(hours=12), pred["dt"].iloc[end] + pd.Timedelta(hours=12),
                           color=RED, alpha=0.16, lw=0)
            start = None
    ax1.plot(pred["dt"], pred["y_pred"], color=BLUE, lw=1.2, label="Logistic regression P(high vol)")
    ax1.plot(base["dt"], base["y_pred"], color=MUTED, lw=0.9, alpha=0.8, label="Baseline percentile score")
    ax1.set_ylim(0, 1.3)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_ylabel("Predicted score")
    ax1.set_title("Out of sample regime forecast, SPY (shaded: realized high volatility regime)")
    ax1.legend(loc="upper right", ncol=2)
    ax2.plot(pred["dt"], pred["vol_21d"], color=AMBER, lw=1.3)
    ax2.set_ylabel("Trailing 21d vol, annualized")
    pct_axis(ax2)
    for b in folds["test_start"].iloc[1:]:
        for ax in (ax1, ax2):
            ax.axvline(pd.Timestamp(b), color=TEXT, lw=0.8, ls=":", alpha=0.5)
    save(fig, "ml_regime.png")

    feat = vr.load_feature_frame(vr._resolve_instrument_id("SPY", None)[0])
    X, y = vr.build_xy(feat)
    pipe = vr._make_model(42)
    pipe.fit(X.to_numpy(float), y.to_numpy(int))
    clf: LogisticRegression = pipe.named_steps["clf"]
    coef = pd.Series(clf.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.barh(coef.index, coef.values, color=[GREEN if v > 0 else RED for v in coef.values])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("Logistic regression coefficients (fit on all data, for interpretation only)")
    ax.grid(axis="y", visible=False)
    save(fig, "ml_coefficients.png")
    return summary


def main() -> None:
    style()
    panels = {
        "equal_universe": q.load_portfolio_panel(1),
        "manual_tilt": q.load_portfolio_panel(2),
    }
    stats = performance(panels)
    correlation(panels["equal_universe"][0])

    returns, weights = panels["equal_universe"]
    perf = q.portfolio_nav_drawdown(returns, weights)
    losses = -perf["ret"]

    boot = bootstrap_distribution(losses.to_numpy())
    var_es(q.load_latest_risk_runs(1), boot)
    kup = backtest(losses)
    summary = ml_figures()

    print("\n--- stats ---")
    for k, v in stats.items():
        print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
    print("bootstrap", boot)
    print("kupiec", kup)
    print("ml baseline", summary["baseline"], "model", summary["model"], "lift", summary["auc_lift"])

    for name, (r, w) in panels.items():
        R = r.to_numpy(float)
        wv = w.to_numpy(float)
        mu, cov = estimate_mean_cov(R)
        mu_p, sig_p = portfolio_mean_std(mu, cov, wv)
        base_var = parametric_var_gaussian(mu_p, sig_p, 0.95)
        stressed = parametric_var_gaussian(mu_p, sig_p * 2.0, 0.95)
        print(name, "parametric VaR", base_var, "vol x2", stressed, "kupiec", kupiec_pof_test(-(R @ wv)[250:], rolling_historical_var(-(R @ wv), 250)[250:], 0.95))


if __name__ == "__main__":
    main()
