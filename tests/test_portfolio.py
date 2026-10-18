import numpy as np
import pytest

from src.portfolio.portfolio_returns import portfolio_losses, portfolio_returns
from src.portfolio.weights import validate_weights


def test_validate_weights_ok():
    w = validate_weights(np.array([0.4, 0.6]))
    assert np.isclose(w.sum(), 1.0)


def test_validate_weights_sum_fail():
    with pytest.raises(ValueError):
        validate_weights(np.array([0.5, 0.6]))


def test_validate_weights_negative_fail():
    with pytest.raises(ValueError):
        validate_weights(np.array([-0.1, 1.1]))


def test_portfolio_returns_manual():
    # Day0: 0.4*0.01 + 0.6*(-0.02) = -0.008
    # Day1: 0.4*0.03 + 0.6*0.01 = 0.018
    R = np.array([[0.01, -0.02], [0.03, 0.01]])
    w = np.array([0.4, 0.6])
    rp = portfolio_returns(R, w)
    assert np.allclose(rp, [-0.008, 0.018])


def test_portfolio_losses_sign():
    R = np.array([[0.01, -0.02]])
    w = np.array([0.4, 0.6])
    L = portfolio_losses(R, w)
    assert np.allclose(L, [0.008])  # -(-0.008)
