"""Audit log: every user-initiated mutation writes one event in the same transaction as the change."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.core.auth import Principal
from skf_api.core.db import utcnow
from skf_api.core.pagination import Keyset, PageParams
from skf_api.modules.audit import schemas
from skf_api.modules.audit.models import AuditEvent

# Keys never persisted in `detail`, whatever a caller passes (the web posts user-management details).
_SECRET_KEYS = {
    "password",
    "new_password",
    "secret",
    "token",
    "private_key",
    "ssh_private_key",
    "api_key",
    "aws_secret_access_key",
    "aws_session_token",
}


@dataclass(frozen=True)
class Actor:
    id: str
    name: str
    role: str

    @classmethod
    def of(cls, principal: Principal) -> Actor:
        return cls(principal.id, principal.name, principal.role)


SYSTEM_ACTOR = Actor(id="system", name="System", role="system")


def _scrub(detail: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in detail.items():
        if key.lower() in _SECRET_KEYS:
            continue
        clean[key] = _scrub(value) if isinstance(value, dict) else value
    return clean


def record(
    session: AsyncSession,
    actor: Actor,
    action: str,
    entity_type: str,
    entity_id: object | None,
    detail: dict[str, Any] | None = None,
    ip: str | None = None,
) -> AuditEvent:
    event = AuditEvent(
        actor_id=actor.id,
        actor_name=actor.name,
        actor_role=actor.role,
        action=action,
        entity_type=entity_type,
        entity_id=None if entity_id is None else str(entity_id),
        detail=_scrub(detail or {}),
        ip=ip,
        ts=utcnow(),
    )
    session.add(event)
    return event


def to_schema(event: AuditEvent) -> schemas.AuditEvent:
    return schemas.AuditEvent(
        id=event.id,
        ts=event.ts,
        actor=schemas.AuditActor(id=event.actor_id, name=event.actor_name, role=event.actor_role),
        action=event.action,
        entity_type=event.entity_type,
        entity_id=event.entity_id,
        detail=event.detail,
    )


_KEYSET = Keyset(AuditEvent.id)


async def list_events(
    session: AsyncSession,
    page: PageParams,
    *,
    actor_id: str | None = None,
    action: str | None = None,
    entity_type: str | None = None,
) -> schemas.AuditEventPage:
    stmt = sa.select(AuditEvent)
    if actor_id:
        stmt = stmt.where(AuditEvent.actor_id == actor_id)
    if action:
        stmt = stmt.where(AuditEvent.action == action)
    if entity_type:
        stmt = stmt.where(AuditEvent.entity_type == entity_type)
    rows = (await session.scalars(_KEYSET.apply(stmt, page))).all()
    return schemas.AuditEventPage(
        items=[to_schema(e) for e in rows[: page.limit]],
        next_cursor=_KEYSET.next_cursor(rows, page, lambda e: (e.id,)),
    )


async def create_event(
    session: AsyncSession, principal: Principal, body: schemas.AuditEventCreate, ip: str | None
) -> schemas.AuditEvent:
    event = record(
        session, Actor.of(principal), body.action, body.entity_type, body.entity_id, body.detail, ip
    )
    await session.commit()
    return to_schema(event)
