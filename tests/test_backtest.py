import numpy as np

from src.risk.backtest import (
    exception_indicator,
    kupiec_pof_test,
    rolling_historical_var,
)


def test_exceptions_scalar_var():
    L = np.array([0.01, 0.02, 0.05, 0.03])
    hits = exception_indicator(L, 0.03)
    assert np.array_equal(hits, [0, 0, 1, 0])


def test_kupiec_well_calibrated_should_not_reject():
    # ~5% hits in large n → should usually fail to reject at 5%
    rng = np.random.default_rng(0)
    n = 2000
    p = 0.05
    hits = rng.random(n) < p
    losses = np.where(hits, 1.0, 0.0)
    var = 0.5
    out = kupiec_pof_test(losses, var, alpha=0.95)
    assert out["reject_5pct"] is False
    assert abs(out["hit_ratio"] - 0.05) < 0.02


def test_kupiec_bad_calibration_should_reject():
    # 20% hits when expect 5% → reject
    n = 1000
    losses = np.zeros(n)
    losses[:200] = 1.0  # 20% exceptions vs var=0.5
    out = kupiec_pof_test(losses, 0.5, alpha=0.95)
    assert out["reject_5pct"] is True
    assert out["p_value"] < 0.05


def test_rolling_var_uses_past_only():
    losses = np.arange(1, 11, dtype=float)  # 1..10
    window = 5
    var_series = rolling_historical_var(losses, window, alpha=0.8)
    assert np.all(np.isnan(var_series[:window]))
    # At t=5, window is losses[0:5] = 1,2,3,4,5 — not including 6
    assert np.isclose(var_series[5], np.quantile(losses[0:5], 0.8))
