"""Typed configuration from the environment (see docs/webapp/SPEC.md §3)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/src/skf_api/settings.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://skf:skf@localhost:55432/skf"
    redis_url: str = "redis://localhost:56379/0"

    auth_jwks_url: str = "http://localhost:3100/api/auth/jwks"
    auth_issuer: str = "skf-skill-studio"
    auth_audience: str = "skf-api"

    secret_key: SecretStr = Field(min_length=32)

    s3_endpoint_url: str | None = None
    s3_public_endpoint_url: str | None = None
    s3_bucket: str = "skf-artifacts"
    s3_region: str = "eu-north-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: SecretStr | None = None

    public_ingest_url: str | None = None
    internal_ingest_url: str = "http://localhost:8000/ingest/v1"

    pipeline_repo_dir: Path = REPO_ROOT
    pipeline_python: str = str(REPO_ROOT / ".venv" / "bin" / "python")
    work_dir: Path = REPO_ROOT / ".skf-work"

    log_max_lines_per_run: int = 200_000
    # Commit the pipeline code was built from, for images that ship without .git (set by the Dockerfiles).
    git_sha: str | None = None

    @field_validator(
        "s3_endpoint_url",
        "s3_public_endpoint_url",
        "s3_access_key_id",
        "s3_secret_access_key",
        "public_ingest_url",
        "git_sha",
        mode="before",
    )
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def permissions_file(self) -> Path:
        in_repo = self.pipeline_repo_dir / "shared" / "permissions.json"
        return in_repo if in_repo.is_file() else REPO_ROOT / "shared" / "permissions.json"

    @property
    def skills_dir(self) -> Path:
        return self.pipeline_repo_dir / "skills"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # values come from the environment
