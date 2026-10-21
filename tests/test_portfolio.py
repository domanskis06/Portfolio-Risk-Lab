import numpy as np
import pytest
from sqlalchemy import text

from src.db.connection import get_engine
from src.portfolio.portfolio_returns import portfolio_losses, portfolio_returns
from src.portfolio.weights import equal_weights, validate_weights


def test_validate_weights_ok():
    w = validate_weights(np.array([0.4, 0.6]))
    assert np.isclose(w.sum(), 1.0)


def test_validate_weights_sum_fail():
    with pytest.raises(ValueError):
        validate_weights(np.array([0.5, 0.6]))


def test_validate_weights_negative_fail():
    with pytest.raises(ValueError):
        validate_weights(np.array([-0.1, 1.1]))


def test_equal_weights_sum_and_index():
    w = equal_weights(["AAA", "BBB", "CCC"])
    assert list(w.index) == ["AAA", "BBB", "CCC"]
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w.to_numpy(), [1 / 3, 1 / 3, 1 / 3])


def test_portfolio_returns_manual():
    # Day0: 0.4*0.01 + 0.6*(-0.02) = -0.008
    # Day1: 0.4*0.03 + 0.6*0.01 = 0.018
    R = np.array([[0.01, -0.02], [0.03, 0.01]])
    w = np.array([0.4, 0.6])
    rp = portfolio_returns(R, w)
    assert np.allclose(rp, [-0.008, 0.018])


def test_portfolio_variance_2x2_on_paper():
    """σ_p^2 = w^T Σ w on a tiny 2x2 example."""
    from src.risk.covariance import portfolio_variance

    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.5, 0.5])
    # w1²σ11 + w2²σ22 + 2 w1 w2 σ12
    # = 0.25*0.04 + 0.25*0.09 + 0.5*0.01 = 0.01 + 0.0225 + 0.005 = 0.0375
    assert np.isclose(portfolio_variance(w, cov), 0.0375)


def test_portfolio_losses_sign():
    R = np.array([[0.01, -0.02]])
    w = np.array([0.4, 0.6])
    L = portfolio_losses(R, w)
    assert np.allclose(L, [0.008])  # -(-0.008)


def test_seed_default_portfolios_idempotent():
    """Requires Docker DB + instruments from ETL/cleaning."""
    from src.etl.clean_prices import ensure_instruments
    from src.portfolio.seed_portfolios import seed_default_portfolios

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — skip when Postgres is down
        pytest.skip(f"Postgres unavailable: {exc}")

    tickers = [
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
    ensure_instruments(tickers)

    ids1 = seed_default_portfolios(tickers)
    ids2 = seed_default_portfolios(tickers)
    assert ids1 == ids2
    assert set(ids1) == {"equal_universe", "manual_tilt"}

    engine = get_engine()
    with engine.connect() as conn:
        n_equal = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM portfolio_weights
                WHERE portfolio_id = :pid
                """
            ),
            {"pid": ids1["equal_universe"]},
        ).scalar_one()
        n_manual = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM portfolio_weights
                WHERE portfolio_id = :pid
                """
            ),
            {"pid": ids1["manual_tilt"]},
        ).scalar_one()
        w_sum = conn.execute(
            text(
                """
                SELECT SUM(weight) FROM portfolio_weights
                WHERE portfolio_id = :pid
                """
            ),
            {"pid": ids1["equal_universe"]},
        ).scalar_one()

    assert n_equal == len(tickers)
    assert n_manual == len(tickers)
    assert np.isclose(float(w_sum), 1.0)
