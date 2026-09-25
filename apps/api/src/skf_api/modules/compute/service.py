"""Compute targets: CRUD, write-only secrets, health checks, and 7-day usage against the weekly quota."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import jsonschema
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.backends.base import BackendError, BackendKind, HealthStatus
from skf_api.core.crypto import SecretBox
from skf_api.core.db import utcnow
from skf_api.core.errors import Conflict, Invalid, NotFound
from skf_api.modules.audit import service as audit
from skf_api.modules.compute import schemas
from skf_api.modules.compute.backends import KINDS_NEEDING_SECRET, BackendFactory
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.runs.models import ACTIVE_STAGE_STATUSES, Stage

log = logging.getLogger(__name__)

USAGE_WINDOW = timedelta(days=7)
# Health `details` key a backend sets when the provider reports its own remaining GPU hours (Kaggle's quota).
PROVIDER_QUOTA_LEFT = "gpu_remaining_hours"


def provider_quota_left(target: ComputeTarget) -> float | None:
    """GPU hours left as the provider reported them at the last health check."""
    value = ((target.health or {}).get("details") or {}).get(PROVIDER_QUOTA_LEFT)
    return float(value) if isinstance(value, int | float) else None


@dataclass(frozen=True)
class Usage:
    gpu_hours_7d: float = 0.0
    cost_7d: float = 0.0
    active_stages: int = 0

    def quota(self, target: ComputeTarget) -> tuple[float, schemas.QuotaSource] | None:
        """Weekly GPU hours left: the lower of the studio's ledger (quota minus its own 7-day usage) and the
        provider's figure, which also counts usage the studio never saw (notebooks, other tools)."""
        left: list[tuple[float, schemas.QuotaSource]] = []
        if target.weekly_quota_gpu_hours is not None:
            left.append((max(0.0, target.weekly_quota_gpu_hours - self.gpu_hours_7d), "ledger"))
        reported = provider_quota_left(target)
        if reported is not None:
            left.append((max(0.0, reported), "provider"))
        return min(left, key=lambda item: item[0]) if left else None

    def quota_left(self, target: ComputeTarget) -> float | None:
        quota = self.quota(target)
        return quota[0] if quota else None

    def to_schema(self, target: ComputeTarget) -> schemas.TargetUsage:
        quota = self.quota(target)
        return schemas.TargetUsage(
            gpu_hours_7d=round(self.gpu_hours_7d, 3),
            cost_7d=round(self.cost_7d, 2),
            active_stages=self.active_stages,
            quota_left_hours=round(quota[0], 2) if quota else None,
            quota_source=quota[1] if quota else None,
        )


async def usage_by_target(session: AsyncSession, target_ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, Usage]:
    """GPU time counts in the window a stage finished in (still-running stages count now)."""
    ids = list(target_ids)
    if not ids:
        return {}
    since = utcnow() - USAGE_WINDOW
    in_window = sa.func.coalesce(Stage.finished_at, sa.func.now()) >= since
    rows = await session.execute(
        sa.select(
            Stage.compute_target_id,
            sa.func.coalesce(sa.func.sum(Stage.gpu_seconds).filter(in_window), 0.0),
            sa.func.coalesce(sa.func.sum(Stage.cost).filter(in_window), 0.0),
            sa.func.count().filter(Stage.status.in_(ACTIVE_STAGE_STATUSES)),
        )
        .where(Stage.compute_target_id.in_(ids))
        .group_by(Stage.compute_target_id)
    )
    return {tid: Usage(gpu_s / 3600.0, cost, active) for tid, gpu_s, cost, active in rows if tid is not None}


def to_schema(target: ComputeTarget, usage: Usage) -> schemas.ComputeTarget:
    health = target.health or {}
    return schemas.ComputeTarget(
        id=target.id,
        name=target.name,
        kind=target.kind,
        description=target.description,
        enabled=target.enabled,
        config=target.config,
        has_secret=target.has_secret,
        gpu_label=target.gpu_label,
        steps_per_second=target.steps_per_second,
        overhead_minutes=target.overhead_minutes,
        cost_per_gpu_hour=target.cost_per_gpu_hour,
        weekly_quota_gpu_hours=target.weekly_quota_gpu_hours,
        max_unapproved_gpu_hours=target.max_unapproved_gpu_hours,
        max_concurrent=target.max_concurrent,
        health=schemas.TargetHealth(
            status=health.get("status", HealthStatus.UNKNOWN.value),
            message=health.get("message", "Not checked yet"),
            checked_at=target.health_checked_at,
        ),
        usage=usage.to_schema(target),
        created_at=target.created_at,
        updated_at=target.updated_at,
    )


async def get_target(session: AsyncSession, target_id: uuid.UUID) -> ComputeTarget:
    target = await session.get(ComputeTarget, target_id)
    if target is None:
        raise NotFound("Compute target not found")
    return target


async def target_view(session: AsyncSession, target: ComputeTarget) -> schemas.ComputeTarget:
    return to_schema(target, (await usage_by_target(session, [target.id])).get(target.id, Usage()))


async def list_targets(session: AsyncSession) -> list[schemas.ComputeTarget]:
    targets = (await session.scalars(sa.select(ComputeTarget).order_by(ComputeTarget.name))).all()
    usage = await usage_by_target(session, [t.id for t in targets])
    return [to_schema(t, usage.get(t.id, Usage())) for t in targets]


def _validate(schema: dict[str, Any], value: dict[str, Any], what: str) -> None:
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(value), key=lambda e: list(e.path))
    if errors:
        # Report where and why, never the offending value: it may be a credential.
        details = {"errors": [{"loc": [what, *map(str, e.path)], "msg": e.validator} for e in errors]}
        raise Invalid(
            f"Invalid {what}: "
            + "; ".join(f"{'.'.join(map(str, e.path)) or what} fails '{e.validator}'" for e in errors),
            details=details,
        )


def _check_enable(backends: BackendFactory, target: ComputeTarget) -> None:
    """An enabled target must be launchable: valid config (seeded targets may lack e.g. a username) and,
    for remote kinds, credentials."""
    if not target.enabled:
        return
    _validate(backends.schema(target.kind)["config"], target.config, "config")
    if target.kind in KINDS_NEEDING_SECRET and not target.has_secret:
        raise Invalid(f"Set the {target.kind.value} credentials before enabling this target")


async def _ensure_unique_name(session: AsyncSession, name: str, exclude: uuid.UUID | None = None) -> None:
    stmt = sa.select(ComputeTarget.id).where(ComputeTarget.name == name)
    if exclude is not None:
        stmt = stmt.where(ComputeTarget.id != exclude)
    if await session.scalar(stmt) is not None:
        raise Conflict(f"A compute target named '{name}' already exists")


async def create_target(
    session: AsyncSession,
    backends: BackendFactory,
    body: schemas.ComputeTargetCreate,
    actor: audit.Actor,
    ip: str | None,
) -> schemas.ComputeTarget:
    await _ensure_unique_name(session, body.name)
    _validate(backends.schema(body.kind)["config"], body.config, "config")
    target = ComputeTarget(**body.model_dump())
    _check_enable(backends, target)
    session.add(target)
    await session.flush()
    audit.record(
        session,
        actor,
        "target.create",
        "compute_target",
        target.id,
        {"name": target.name, "kind": target.kind.value},
        ip,
    )
    await session.commit()
    return await target_view(session, target)


async def update_target(
    session: AsyncSession,
    backends: BackendFactory,
    target_id: uuid.UUID,
    body: schemas.ComputeTargetUpdate,
    actor: audit.Actor,
    ip: str | None,
) -> schemas.ComputeTarget:
    target = await get_target(session, target_id)
    changes = body.model_dump(exclude_unset=True)
    if changes.get("name") is None:
        changes.pop("name", None)
    else:
        await _ensure_unique_name(session, changes["name"], exclude=target.id)
    if "config" in changes:
        changes["config"] = changes["config"] or {}
        _validate(backends.schema(target.kind)["config"], changes["config"], "config")
    for field, value in changes.items():
        if (
            field
            in (
                "enabled",
                "steps_per_second",
                "overhead_minutes",
                "cost_per_gpu_hour",
                "max_unapproved_gpu_hours",
                "max_concurrent",
            )
            and value is None
        ):
            raise Invalid(f"'{field}' cannot be null")
        setattr(target, field, value)
    _check_enable(backends, target)
    audit.record(
        session, actor, "target.update", "compute_target", target.id, {"fields": sorted(changes)}, ip
    )
    await session.commit()
    return await target_view(session, target)


async def set_secret(
    session: AsyncSession,
    backends: BackendFactory,
    secrets: SecretBox,
    target_id: uuid.UUID,
    secret: dict[str, Any],
    actor: audit.Actor,
    ip: str | None,
) -> None:
    target = await get_target(session, target_id)
    _validate(backends.schema(target.kind)["secret"], secret, "secret")
    target.secret_enc = secrets.encrypt_json(secret)
    target.updated_at = utcnow()
    audit.record(
        session, actor, "target.secret_set", "compute_target", target.id, {"fields": sorted(secret)}, ip
    )
    await session.commit()


async def check_health(
    session: AsyncSession, backends: BackendFactory, target_id: uuid.UUID, actor: audit.Actor, ip: str | None
) -> schemas.ComputeTarget:
    target = await get_target(session, target_id)
    try:
        report = await backends.for_target(target).validate()
        health = {"status": report.status.value, "message": report.message, "details": report.details}
    except BackendError as exc:
        health = {"status": HealthStatus.DOWN.value, "message": str(exc), "details": {}}
    except Exception as exc:
        log.exception("health check failed", extra={"target_id": str(target.id)})
        health = {
            "status": HealthStatus.DOWN.value,
            "message": f"Health check failed: {type(exc).__name__}",
            "details": {},
        }
    target.health = health
    target.health_checked_at = utcnow()
    audit.record(
        session, actor, "target.check", "compute_target", target.id, {"status": health["status"]}, ip
    )
    await session.commit()
    return await target_view(session, target)


def backend_schema(backends: BackendFactory, kind: BackendKind) -> schemas.BackendSchema:
    schema = backends.schema(kind)
    return schemas.BackendSchema(config=schema["config"], secret=schema["secret"])
