import numpy as np

from src.risk.covariance import portfolio_mean_std
from src.risk.var_montecarlo import monte_carlo_var_es, simulate_multivariate_normal
from src.risk.var_parametric import parametric_var_gaussian


def test_simulate_shape_and_seed():
    mu = np.zeros(2)
    cov = np.eye(2) * 0.01
    a = simulate_multivariate_normal(mu, cov, n_sims=50, seed=7)
    b = simulate_multivariate_normal(mu, cov, n_sims=50, seed=7)
    assert a.shape == (50, 2)
    assert np.allclose(a, b)


def test_mc_close_to_parametric_large_n():
    # Pure Gaussian world: MC VaR should ≈ parametric VaR
    mu = np.array([0.0, 0.0])
    cov = np.array([[0.0004, 0.0], [0.0, 0.0004]])  # sigma_i = 0.02
    w = np.array([0.5, 0.5])
    alpha = 0.95
    mu_p, sigma_p = portfolio_mean_std(mu, cov, w)
    var_p = parametric_var_gaussian(mu_p, sigma_p, alpha)
    var_mc, es_mc = monte_carlo_var_es(mu, cov, w, alpha=alpha, n_sims=500_000, seed=0)
    assert np.isclose(var_mc, var_p, rtol=0.05)  # ~5% relative tol
    assert es_mc >= var_mc
