"""Smoke tests for project config (should pass after venv + pip install)."""

from __future__ import annotations

from src import config


def test_project_root_exists() -> None:
    assert config.PROJECT_ROOT.is_dir()


def test_sql_dir_has_schema() -> None:
    schema = config.SQL_DIR / "001_schema.sql"
    assert schema.is_file()


def test_random_seed_is_int() -> None:
    assert isinstance(config.RANDOM_SEED, int)
