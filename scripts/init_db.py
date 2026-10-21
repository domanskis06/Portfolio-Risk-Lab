"""Apply sql/001_schema.sql to DATABASE_URL and list the resulting tables.

Usage:
    python scripts/init_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from src.config import SQL_DIR
from src.db.connection import get_engine


def main() -> None:
    schema_path = SQL_DIR / "001_schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql))

    print(f"Applied schema: {schema_path}")
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        ).fetchall()
    print("Tables:", ", ".join(r[0] for r in rows))


if __name__ == "__main__":
    main()
