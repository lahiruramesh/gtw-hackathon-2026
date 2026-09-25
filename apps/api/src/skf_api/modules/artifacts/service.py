"""Artifact listing, classification of job outputs, and presigned downloads."""

from __future__ import annotations

import mimetypes
import re
import uuid
from collections.abc import Iterable
from pathlib import PurePosixPath

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.errors import NotFound
from skf_api.core.storage import Storage
from skf_api.modules.artifacts import schemas
from skf_api.modules.artifacts.models import Artifact, ArtifactKind
from skf_api.modules.runs.models import Stage

_CKPT = re.compile(r"^ckpt_(\d+)\.pkl$")
_BY_SUFFIX = {
    ".mp4": ArtifactKind.VIDEO,
    ".webm": ArtifactKind.VIDEO,
    ".gif": ArtifactKind.IMAGE,
    ".png": ArtifactKind.IMAGE,
    ".jpg": ArtifactKind.IMAGE,
    ".jpeg": ArtifactKind.IMAGE,
    ".svg": ArtifactKind.IMAGE,
    ".csv": ArtifactKind.CSV,
    ".json": ArtifactKind.JSON,
    ".log": ArtifactKind.LOG,
    ".txt": ArtifactKind.LOG,
    ".yaml": ArtifactKind.CONFIG,
    ".yml": ArtifactKind.CONFIG,
}
CHECKPOINT_KINDS = (ArtifactKind.CHECKPOINT, ArtifactKind.PARAMS)


def classify(name: str) -> tuple[ArtifactKind, int | None]:
    """Kind and training step of a job output, from its relative path."""
    base = PurePosixPath(name).name
    if base == "params.pkl":
        return ArtifactKind.PARAMS, None
    if match := _CKPT.match(base):
        return ArtifactKind.CHECKPOINT, int(match.group(1))
    if base == "config.json":
        return ArtifactKind.CONFIG, None
    return _BY_SUFFIX.get(PurePosixPath(base).suffix.lower(), ArtifactKind.OTHER), None


def content_type(name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def to_schema(artifact: Artifact, stage_key: str | None) -> schemas.Artifact:
    return schemas.Artifact(
        id=artifact.id,
        stage_id=artifact.stage_id,
        stage_key=stage_key,
        kind=artifact.kind,
        name=artifact.name,
        size_bytes=artifact.size_bytes,
        content_type=artifact.content_type,
        step=artifact.step,
        created_at=artifact.created_at,
    )


def _with_stage_key() -> sa.Select[Artifact, str]:
    return sa.select(Artifact, Stage.key).outerjoin(Stage, Stage.id == Artifact.stage_id)


async def artifacts_by_id(
    session: AsyncSession, ids: Iterable[uuid.UUID]
) -> dict[uuid.UUID, schemas.Artifact]:
    wanted = [i for i in set(ids) if i is not None]
    if not wanted:
        return {}
    rows = await session.execute(_with_stage_key().where(Artifact.id.in_(wanted)))
    return {a.id: to_schema(a, key) for a, key in rows}


async def list_for_run(session: AsyncSession, run_id: uuid.UUID) -> list[schemas.Artifact]:
    rows = await session.execute(
        _with_stage_key()
        .where(Artifact.run_id == run_id)
        .order_by(Stage.position.nulls_last(), Artifact.kind, Artifact.step.nulls_last(), Artifact.name)
    )
    return [to_schema(a, key) for a, key in rows]


async def presigned_url(
    session: AsyncSession, storage: Storage, artifact_id: uuid.UUID
) -> schemas.ArtifactUrl:
    artifact = await session.get(Artifact, artifact_id)
    if artifact is None:
        raise NotFound("Artifact not found")
    signed = await storage.presigned_get(
        artifact.uri, filename=PurePosixPath(artifact.name).name, content_type=artifact.content_type
    )
    return schemas.ArtifactUrl(url=signed.url, expires_at=signed.expires_at)
