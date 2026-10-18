import numpy as np

from src.risk.stress import apply_parallel_shock, scale_cov_vol


def test_parallel_shock_loss():
    w = np.array([0.5, 0.5])
    assert np.isclose(apply_parallel_shock(w, shock=-0.10), 0.10)


def test_scale_cov_vol():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    out = scale_cov_vol(cov, vol_mult=2.0)
    assert np.allclose(out, cov * 4.0)
