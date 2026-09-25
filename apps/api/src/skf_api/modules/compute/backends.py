"""Builds ComputeBackend instances for compute_targets rows (the only place secrets are decrypted)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Protocol

from skf_api.backends import config_schema, create_backend
from skf_api.backends.base import BackendKind, ComputeBackend, TargetContext
from skf_api.core.crypto import SecretBox
from skf_api.modules.compute.models import ComputeTarget
from skf_api.settings import Settings

KINDS_NEEDING_SECRET = frozenset({BackendKind.KAGGLE, BackendKind.AWS_EC2})


class BackendFactory(Protocol):
    def for_target(self, target: ComputeTarget) -> ComputeBackend: ...
    def schema(self, kind: BackendKind) -> dict[str, Any]: ...


class DefaultBackendFactory:
    """Caches one backend per target version: the local backend supervises processes in-process, and
    rebuilding on every reconcile tick would drop that state. A config/secret change bumps updated_at."""

    def __init__(self, settings: Settings, secrets: SecretBox):
        self._settings = settings
        self._secrets = secrets
        self._cache: dict[uuid.UUID, tuple[datetime, ComputeBackend]] = {}

    def for_target(self, target: ComputeTarget) -> ComputeBackend:
        cached = self._cache.get(target.id)
        if cached and cached[0] == target.updated_at:
            return cached[1]
        ctx = TargetContext(
            target_id=str(target.id),
            name=target.name,
            config=dict(target.config),
            secret=self._secrets.decrypt_json(target.secret_enc) if target.secret_enc else {},
            work_dir=self._settings.work_dir / "targets" / str(target.id),
            pipeline_repo_dir=self._settings.pipeline_repo_dir,
            pipeline_python=self._settings.pipeline_python,
        )
        backend = create_backend(target.kind, ctx)
        self._cache[target.id] = (target.updated_at, backend)
        return backend

    def schema(self, kind: BackendKind) -> dict[str, Any]:
        return config_schema(kind)
