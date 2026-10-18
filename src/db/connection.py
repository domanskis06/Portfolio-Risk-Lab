"""SQLAlchemy engine / connection helpers.

TODO (Step 1 — you implement):
- create_engine(DATABASE_URL)
- get_connection() context manager or session factory
- optional: ping() that runs SELECT 1
"""

from __future__ import annotations
from src.config import DATABASE_URL
from sqlalchemy import create_engine, Engine


def get_engine() -> Engine:
    """
    Return a SQLAlchemy Engine for DATABASE_URL.
    """
    return create_engine(DATABASE_URL, pool_pre_ping=True)