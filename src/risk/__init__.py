"""Portfolio risk: covariance, VaR, ES, Monte Carlo, backtest, stress."""

from src.risk.backtest import (
    exception_indicator,
    kupiec_pof_test,
    rolling_historical_var,
    rolling_kupiec,
)
from src.risk.covariance import (
    estimate_mean_cov,
    portfolio_mean_std,
    portfolio_variance,
    sample_covariance,
)
from src.risk.expected_shortfall import expected_shortfall, historical_es
from src.risk.historical import historical_var
from src.risk.montecarlo import (
    monte_carlo_es,
    monte_carlo_var,
    monte_carlo_var_es,
    simulate_multivariate_normal,
)
from src.risk.parametric import (
    parametric_es,
    parametric_es_gaussian,
    parametric_var,
    parametric_var_gaussian,
)
from src.risk.stress import (
    apply_parallel_shock,
    scale_cov_vol,
    stressed_portfolio_loss_parallel,
)

__all__ = [
    "historical_var",
    "historical_es",
    "expected_shortfall",
    "estimate_mean_cov",
    "portfolio_mean_std",
    "sample_covariance",
    "portfolio_variance",
    "parametric_var_gaussian",
    "parametric_es_gaussian",
    "parametric_var",
    "parametric_es",
    "simulate_multivariate_normal",
    "monte_carlo_var_es",
    "monte_carlo_var",
    "monte_carlo_es",
    "exception_indicator",
    "kupiec_pof_test",
    "rolling_historical_var",
    "rolling_kupiec",
    "apply_parallel_shock",
    "scale_cov_vol",
    "stressed_portfolio_loss_parallel",
]
