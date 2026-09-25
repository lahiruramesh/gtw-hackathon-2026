"""Where the api-core tests find their real services, overridable with TEST_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from alembic.config import Config

API_DIR = Path(__file__).resolve().parents[1]
DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://skf:skf@localhost:55432/skf_test")
REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:56379/15")
S3_ENDPOINT_URL = os.environ.get("TEST_S3_ENDPOINT_URL", "http://localhost:59000")


def alembic_config() -> Config:
    config = Config(str(API_DIR / "alembic.ini"))
    config.attributes["database_url"] = DATABASE_URL
    config.attributes["configure_logger"] = False
    return config
