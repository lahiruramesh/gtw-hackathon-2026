"""Process-wide resources shared by the API and the worker, built once at startup."""

from __future__ import annotations

from dataclasses import dataclass

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from skf_api.core.auth import TokenVerifier
from skf_api.core.crypto import SecretBox
from skf_api.core.db import Database
from skf_api.core.events import EventBus
from skf_api.core.ingest_tokens import IngestTokenSigner
from skf_api.core.jobs import ArqJobQueue, JobQueue
from skf_api.core.permissions import PermissionTable
from skf_api.core.storage import S3Config, S3Storage, Storage
from skf_api.modules.compute.backends import BackendFactory, DefaultBackendFactory
from skf_api.settings import Settings
from skf_api.skills_registry.registry import SkillRegistry


@dataclass
class AppContext:
    settings: Settings
    db: Database
    redis: ArqRedis
    storage: Storage
    events: EventBus
    jobs: JobQueue
    permissions: PermissionTable
    verifier: TokenVerifier
    secrets: SecretBox
    ingest_tokens: IngestTokenSigner
    registry: SkillRegistry
    backends: BackendFactory

    async def close(self) -> None:
        await self.redis.aclose()
        await self.db.dispose()


def storage_from_settings(settings: Settings) -> S3Storage:
    return S3Storage(
        S3Config(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            public_endpoint_url=settings.s3_public_endpoint_url,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key.get_secret_value()
            if settings.s3_secret_access_key
            else None,
        )
    )


async def build_context(
    settings: Settings,
    *,
    verifier: TokenVerifier | None = None,
    jobs: JobQueue | None = None,
    backends: BackendFactory | None = None,
) -> AppContext:
    """`verifier`, `jobs` and `backends` are injectable for tests (local JWKS, recorded jobs, fakes)."""
    permissions = PermissionTable.load(settings.permissions_file)
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    secret_key = settings.secret_key.get_secret_value()
    secrets = SecretBox(secret_key)
    registry = SkillRegistry(settings.skills_dir, fallback_git_sha=settings.git_sha)
    registry.reload()
    return AppContext(
        settings=settings,
        db=Database(settings.database_url),
        redis=redis,
        storage=storage_from_settings(settings),
        events=EventBus(redis),
        jobs=jobs or ArqJobQueue(redis),
        permissions=permissions,
        verifier=verifier
        or TokenVerifier.from_jwks_url(
            settings.auth_jwks_url,
            issuer=settings.auth_issuer,
            audience=settings.auth_audience,
            permissions=permissions,
        ),
        secrets=secrets,
        ingest_tokens=IngestTokenSigner(secret_key),
        registry=registry,
        backends=backends or DefaultBackendFactory(settings, secrets),
    )
