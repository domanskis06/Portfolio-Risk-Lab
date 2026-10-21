"""SQLAlchemy engine factory."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine

from src.config import DATABASE_URL


def get_engine() -> Engine:
    """Return a SQLAlchemy Engine for DATABASE_URL."""
    return create_engine(DATABASE_URL, pool_pre_ping=True)
