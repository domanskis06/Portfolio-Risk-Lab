"""Step 1: make this pass by implementing get_engine() and having Docker Postgres up."""

from __future__ import annotations

from sqlalchemy import text


def test_database_ping() -> None:
    """Test that the database is reachable."""
    from src.db.connection import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        value = conn.execute(text("SELECT 1")).scalar_one()
    assert value == 1
