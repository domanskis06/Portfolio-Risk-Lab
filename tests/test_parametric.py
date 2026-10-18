import numpy as np
from scipy.stats import norm

from src.risk.covariance import estimate_mean_cov, portfolio_mean_std
from src.risk.var_parametric import parametric_var_gaussian


def test_parametric_var_zero_mean():
    # mu_p=0, sigma_p=0.02, alpha=0.95 → VaR = 0.02 * Φ^{-1}(0.95)
    sigma = 0.02
    alpha = 0.95
    expected = sigma * norm.ppf(alpha)
    assert np.isclose(parametric_var_gaussian(0.0, sigma, alpha), expected)


def test_portfolio_variance_2x2():
    # w=[0.5,0.5], Sigma = [[0.04, 0.01], [0.01, 0.09]]
    # w'Σw = 0.25*0.04 + 0.25*0.09 + 2*0.5*0.5*0.01 = 0.01 + 0.0225 + 0.005 = 0.0375
    mu = np.array([0.0, 0.0])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.5, 0.5])
    mu_p, sigma_p = portfolio_mean_std(mu, cov, w)
    assert np.isclose(mu_p, 0.0)
    assert np.isclose(sigma_p**2, 0.0375)


def test_estimate_mean_cov_shapes():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(100, 3))
    mu, cov = estimate_mean_cov(R)
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)
