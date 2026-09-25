"""api-core test fixtures: real Postgres (`skf_test`), Redis (db 15) and MinIO (`skf-artifacts-test`).

Nothing here is autouse, so tests/backends/ (which needs none of it) is unaffected.
Override the service URLs with TEST_DATABASE_URL / TEST_REDIS_URL / TEST_S3_ENDPOINT_URL.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from api_fakes import AUDIENCE, ISSUER, FakeBackendFactory, FakeWorld, Keys, RecordingJobQueue
from api_services import DATABASE_URL, REDIS_URL, S3_ENDPOINT_URL, alembic_config
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command
from skf_api.context import AppContext, build_context
from skf_api.core.auth import TokenVerifier
from skf_api.core.permissions import PermissionTable
from skf_api.main import create_app
from skf_api.settings import REPO_ROOT, Settings

TABLES = (
    "audit_events",
    "gate_decisions",
    "evaluations",
    "metric_points",
    "log_lines",
    "artifacts",
    "stages",
    "runs",
    "compute_targets",
    "skills",
)


def make_settings(work_dir: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "database_url": DATABASE_URL,
        "redis_url": REDIS_URL,
        "auth_jwks_url": "http://127.0.0.1:1/unused",
        "auth_issuer": ISSUER,
        "auth_audience": AUDIENCE,
        "secret_key": "test-secret-key-that-is-long-enough-0123456789",
        "s3_endpoint_url": S3_ENDPOINT_URL,
        "s3_public_endpoint_url": "http://files.example.test",
        "s3_bucket": "skf-artifacts-test",
        "s3_region": "us-east-1",
        "s3_access_key_id": "skfminio",
        "s3_secret_access_key": "skfminio-dev-secret",
        "public_ingest_url": None,
        "internal_ingest_url": "http://api.test/ingest/v1",
        "pipeline_repo_dir": REPO_ROOT,
        "work_dir": work_dir,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


# ------------------------------------------------------------------------------------------------ JWKS / JWT


@pytest.fixture(scope="session")
def keys() -> Keys:
    return Keys()


@pytest.fixture(scope="session")
def jwks_url(keys: Keys) -> Iterator[str]:
    body = json.dumps(keys.jwks).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api/auth/jwks"
    server.shutdown()


# ------------------------------------------------------------------------------------------------ database


@pytest.fixture(scope="session")
def migrated() -> None:
    """Fresh `app` schema in skf_test, built by the real Alembic migration."""

    async def reset() -> None:
        engine = create_async_engine(DATABASE_URL)
        async with engine.begin() as conn:
            await conn.execute(sa.text("DROP SCHEMA IF EXISTS app CASCADE"))
        await engine.dispose()

    asyncio.run(reset())
    command.upgrade(alembic_config(), "head")


@pytest.fixture
def world() -> FakeWorld:
    return FakeWorld()


@pytest.fixture
def queue() -> RecordingJobQueue:
    return RecordingJobQueue()


@pytest.fixture
async def ctx(
    migrated: None, jwks_url: str, tmp_path: Path, world: FakeWorld, queue: RecordingJobQueue
) -> AsyncIterator[AppContext]:
    settings = make_settings(tmp_path / "work")
    permissions = PermissionTable.load(settings.permissions_file)
    verifier = TokenVerifier.from_jwks_url(
        jwks_url, issuer=ISSUER, audience=AUDIENCE, permissions=permissions
    )
    context = await build_context(settings, verifier=verifier, jobs=queue, backends=FakeBackendFactory(world))
    async with context.db.engine.begin() as conn:
        await conn.execute(sa.text("TRUNCATE " + ", ".join(f"app.{t}" for t in TABLES) + " RESTART IDENTITY"))
    await context.redis.flushdb()
    await context.storage.ensure_bucket()
    try:
        yield context
    finally:
        await context.close()


@pytest.fixture
async def client(ctx: AppContext) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(ctx))
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as http:
        yield http


@pytest.fixture
async def skills(ctx: AppContext) -> list[str]:
    """The repo's real skill manifests (g1-step-length, g1-stairs) in the catalog."""
    from skf_api.modules.skills.service import sync_skills

    async with ctx.db.session() as session:
        result = await sync_skills(session, ctx.registry)
    assert not result.errors, result.errors
    return result.synced


@pytest.fixture
async def targets(ctx: AppContext, skills: list[str]) -> dict[str, uuid.UUID]:
    """Seeded targets, with the remote ones given credentials and enabled (as an admin would)."""
    from skf_api.modules.compute.models import ComputeTarget
    from skf_api.modules.compute.seeds import default_seeds, seed_targets

    async with ctx.db.session() as session:
        await seed_targets(session, default_seeds(kaggle_username="skf-test"))
        rows = (await session.scalars(sa.select(ComputeTarget))).all()
        for target in rows:
            if target.kind.value != "local_cpu":
                target.secret_enc = ctx.secrets.encrypt_json(
                    {"key": "k"} if target.kind.value == "kaggle" else {"ssh_private_key": "pem"}
                )
                target.enabled = True
        await session.commit()
        return {t.name: t.id for t in rows}


@pytest.fixture
def auth(keys: Keys) -> Callable[..., dict[str, str]]:
    return keys.headers
