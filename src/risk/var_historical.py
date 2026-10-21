"""Backward-compatible alias for src.risk.historical."""

from src.risk.historical import historical_es, historical_var

__all__ = ["historical_var", "historical_es"]
