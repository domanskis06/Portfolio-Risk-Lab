"""Backward-compatible alias for src.risk.parametric."""

from src.risk.parametric import (
    parametric_es,
    parametric_es_gaussian,
    parametric_var,
    parametric_var_gaussian,
)

__all__ = [
    "parametric_var_gaussian",
    "parametric_es_gaussian",
    "parametric_var",
    "parametric_es",
]
