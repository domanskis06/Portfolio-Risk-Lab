"""Volatility regime classifier (Option A).

Baseline: threshold on contemporaneous vol_21d (fit quantile on train only).
Model: LogisticRegression or HistGradientBoosting on point-in-time features
(vol term-structure, vol momentum, |ret| — no future info).
Persists aggregated metrics to model_runs and OOS rows to predictions.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.config import RANDOM_SEED
from src.db.connection import get_engine
from src.ml.dataset import make_vol_regime_labels, purged_time_split
from src.ml.evaluate import classification_metrics

ModelKind = Literal["logreg", "hgb"]

# Tuned on SPY walk-forward: richer features + logreg; horizon=42 lifts absolute AUC
# (task is more persistent — baseline rises too). Default keeps h=21 unless overridden.
FEATURE_COLS = [
    "vol_21d",
    "beta_63d",
    "abs_ret_1d",
    "ret_1d",
    "vol_5d",
    "vol_63d",
    "vol_ratio_21_63",
    "vol_chg_5d",
    "vol_lag_5",
    "abs_ret_ma_5",
    "ret_sq_ma_10",
]
DEFAULT_TICKER = "SPY"
DEFAULT_HORIZON = 21
DEFAULT_QUANTILE = 0.7
DEFAULT_N_SPLITS = 5
DEFAULT_MODEL: ModelKind = "logreg"


def _resolve_instrument_id(ticker: str | None, instrument_id: int | None) -> tuple[int, str]:
    if instrument_id is not None:
        with get_engine().connect() as conn:
            row = conn.execute(
                text("SELECT instrument_id, ticker FROM instruments WHERE instrument_id = :id"),
                {"id": int(instrument_id)},
            ).one_or_none()
        if row is None:
            raise ValueError(f"Unknown instrument_id={instrument_id}")
        return int(row[0]), str(row[1])

    ticker = ticker or DEFAULT_TICKER
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT instrument_id, ticker FROM instruments WHERE ticker = :ticker"),
            {"ticker": ticker},
        ).one_or_none()
    if row is None:
        raise ValueError(f"Ticker {ticker!r} not in instruments")
    return int(row[0]), str(row[1])


def load_feature_frame(instrument_id: int) -> pd.DataFrame:
    """Load features for one instrument; index=dt, columns include ret_1d / vol / beta."""
    query = text(
        """
        SELECT dt, ret_1d, vol_21d, beta_63d
        FROM features
        WHERE instrument_id = :instrument_id
        ORDER BY dt
        """
    )
    with get_engine().connect() as conn:
        result = conn.execute(query, {"instrument_id": int(instrument_id)})
        df = pd.DataFrame(
            result.fetchall(),
            columns=["dt", "ret_1d", "vol_21d", "beta_63d"],
        )
    if df.empty:
        raise ValueError(f"No features for instrument_id={instrument_id}")
    df["dt"] = pd.to_datetime(df["dt"])
    return df.set_index("dt").sort_index().astype(float)


def build_xy(
    feat: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    high_vol_quantile: float = DEFAULT_QUANTILE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Point-in-time X and forward-vol labels y; drop rows with any NaN.

    Extra features (all ≤ t): short/long realized vol, vol ratio, 5d vol change,
    lagged vol, short |ret| / r^2 means.
    """
    labels = make_vol_regime_labels(
        feat["ret_1d"], horizon=horizon, high_vol_quantile=high_vol_quantile
    )
    r = feat["ret_1d"]
    vol21 = feat["vol_21d"]
    vol63 = r.rolling(63).std()
    X = pd.DataFrame(
        {
            "vol_21d": vol21,
            "beta_63d": feat["beta_63d"],
            "abs_ret_1d": r.abs(),
            "ret_1d": r,
            "vol_5d": r.rolling(5).std(),
            "vol_63d": vol63,
            "vol_ratio_21_63": vol21 / vol63.replace(0.0, np.nan),
            "vol_chg_5d": vol21.diff(5),
            "vol_lag_5": vol21.shift(5),
            "abs_ret_ma_5": r.abs().rolling(5).mean(),
            "ret_sq_ma_10": (r**2).rolling(10).mean(),
        },
        index=feat.index,
    )
    mask = X.notna().all(axis=1) & labels.notna()
    return X.loc[mask], labels.loc[mask].astype(int)


def _baseline_scores(vol_train: np.ndarray, vol_test: np.ndarray, q: float) -> np.ndarray:
    """Score = 1 if test vol >= train-only quantile threshold, else 0."""
    thresh = float(np.quantile(vol_train, q))
    return (vol_test >= thresh).astype(float)


def _make_model(seed: int, model_kind: ModelKind = DEFAULT_MODEL):
    if model_kind == "hgb":
        # Shallow HGB — deep trees overfit hard on this small walk-forward task.
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            random_state=seed,
            class_weight="balanced",
        )
    if model_kind != "logreg":
        raise ValueError(f"Unknown model_kind={model_kind!r}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    class_weight="balanced",
                    C=0.5,
                ),
            ),
        ]
    )


def _mean_metric(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def insert_model_run(
    model_name: str,
    metrics: dict[str, Any],
    params: dict[str, Any],
    *,
    train_start: Any = None,
    train_end: Any = None,
    test_start: Any = None,
    test_end: Any = None,
) -> int:
    """INSERT model_runs row; return run_id."""
    query = text(
        """
        INSERT INTO model_runs (
            model_name, train_start, train_end, test_start, test_end,
            metrics_json, params_json
        )
        VALUES (
            :model_name, :train_start, :train_end, :test_start, :test_end,
            CAST(:metrics_json AS jsonb), CAST(:params_json AS jsonb)
        )
        RETURNING run_id
        """
    )
    with get_engine().begin() as conn:
        run_id = conn.execute(
            query,
            {
                "model_name": model_name,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "metrics_json": json.dumps(metrics),
                "params_json": json.dumps(params),
            },
        ).scalar_one()
    return int(run_id)


def insert_predictions(
    run_id: int,
    instrument_id: int,
    dates: pd.DatetimeIndex | pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """INSERT prediction rows for one run (hard labels / scores as floats)."""
    query = text(
        """
        INSERT INTO predictions (run_id, dt, instrument_id, y_true, y_pred)
        VALUES (:run_id, :dt, :instrument_id, :y_true, :y_pred)
        """
    )
    rows = [
        {
            "run_id": int(run_id),
            "dt": pd.Timestamp(dt).date(),
            "instrument_id": int(instrument_id),
            "y_true": float(yt),
            "y_pred": float(yp),
        }
        for dt, yt, yp in zip(dates, y_true, y_pred, strict=True)
    ]
    if not rows:
        return
    with get_engine().begin() as conn:
        conn.execute(query, rows)


def train_vol_regime_model(
    ticker: str | None = DEFAULT_TICKER,
    instrument_id: int | None = None,
    *,
    horizon: int = DEFAULT_HORIZON,
    high_vol_quantile: float = DEFAULT_QUANTILE,
    n_splits: int = DEFAULT_N_SPLITS,
    purge_gap: int | None = None,
    model_kind: ModelKind = DEFAULT_MODEL,
    seed: int = RANDOM_SEED,
    persist: bool = True,
) -> dict[str, Any]:
    """Walk-forward train baseline + model; optionally write DB rows.

    Returns a summary dict with metrics and optional run_ids.
    Tip: ``horizon=42, model_kind='hgb'`` often raises absolute AUC on SPY, but
    compare lift vs baseline — the label task also gets easier.
    """
    iid, resolved_ticker = _resolve_instrument_id(ticker, instrument_id)
    gap = horizon if purge_gap is None else int(purge_gap)

    feat = load_feature_frame(iid)
    X, y = build_xy(feat, horizon=horizon, high_vol_quantile=high_vol_quantile)
    if len(X) < (n_splits + 1) * 30:
        raise ValueError(
            f"Too few labeled rows ({len(X)}) for n_splits={n_splits}; "
            "load more history or reduce n_splits"
        )

    splits = purged_time_split(len(X), n_splits=n_splits, purge_gap=gap)
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)
    vol_np = X["vol_21d"].to_numpy(dtype=float)

    baseline_fold_metrics: list[dict] = []
    model_fold_metrics: list[dict] = []
    oos_dates: list[pd.Timestamp] = []
    oos_true: list[float] = []
    oos_pred: list[float] = []  # model P(high)
    oos_base: list[float] = []

    for fold_i, (tr, te) in enumerate(splits):
        y_tr, y_te = y_np[tr], y_np[te]
        if len(np.unique(y_tr)) < 2:
            continue

        base_score = _baseline_scores(vol_np[tr], vol_np[te], high_vol_quantile)
        baseline_fold_metrics.append(classification_metrics(y_te, base_score))

        clf = _make_model(seed, model_kind=model_kind)
        clf.fit(X_np[tr], y_tr)
        proba = clf.predict_proba(X_np[te])[:, 1]
        model_fold_metrics.append(classification_metrics(y_te, proba))

        oos_dates.extend(list(X.index[te]))
        oos_true.extend(y_te.astype(float).tolist())
        oos_pred.extend(proba.tolist())
        oos_base.extend(base_score.tolist())

    base_auc = _mean_metric(baseline_fold_metrics, "roc_auc")
    mod_auc = _mean_metric(model_fold_metrics, "roc_auc")
    summary: dict[str, Any] = {
        "ticker": resolved_ticker,
        "instrument_id": iid,
        "n_rows": int(len(X)),
        "n_folds_used": len(model_fold_metrics),
        "model_kind": model_kind,
        "horizon": horizon,
        "high_vol_quantile": high_vol_quantile,
        "baseline": {
            "roc_auc": base_auc,
            "average_precision": _mean_metric(baseline_fold_metrics, "average_precision"),
        },
        "model": {
            "roc_auc": mod_auc,
            "average_precision": _mean_metric(model_fold_metrics, "average_precision"),
        },
        "auc_lift": (mod_auc - base_auc) if (mod_auc is not None and base_auc is not None) else None,
        "run_ids": {},
    }

    params = {
        "ticker": resolved_ticker,
        "instrument_id": iid,
        "horizon": horizon,
        "high_vol_quantile": high_vol_quantile,
        "n_splits": n_splits,
        "purge_gap": gap,
        "seed": seed,
        "model_kind": model_kind,
        "features": FEATURE_COLS,
        "n_rows": int(len(X)),
    }

    model_name = f"vol_regime_{model_kind}"

    if persist and oos_dates:
        dates = pd.DatetimeIndex(oos_dates)
        train_start = X.index.min().date()
        train_end = X.index.max().date()
        test_start = dates.min().date()
        test_end = dates.max().date()

        base_id = insert_model_run(
            "vol_regime_baseline",
            summary["baseline"],
            {**params, "kind": "vol_quantile_threshold"},
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        insert_predictions(
            base_id, iid, dates, np.asarray(oos_true), np.asarray(oos_base)
        )

        model_id = insert_model_run(
            model_name,
            summary["model"],
            {**params, "kind": model_kind},
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        insert_predictions(
            model_id, iid, dates, np.asarray(oos_true), np.asarray(oos_pred)
        )
        summary["run_ids"] = {"baseline": base_id, "model": model_id}

    return summary
