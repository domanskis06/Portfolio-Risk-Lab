"""Project configuration loaded from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://qrwh:qrwh@localhost:5432/quant_research_warehouse",
)

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_STAGING_DIR = PROJECT_ROOT / "data" / "staging"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SQL_DIR = PROJECT_ROOT / "sql"
