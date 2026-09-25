"""Roles -> permissions, loaded from shared/permissions.json (the single source for api and web)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class Permission:
    """Permission strings exactly as they appear in shared/permissions.json."""

    SKILL_READ = "skill:read"
    SKILL_WRITE = "skill:write"
    RUN_READ = "run:read"
    RUN_CREATE_PRESET = "run:create_preset"
    RUN_CREATE_CUSTOM = "run:create_custom"
    RUN_CANCEL_ANY = "run:cancel_any"
    RUN_APPROVE_LAUNCH = "run:approve_launch"
    RUN_EVALUATE = "run:evaluate"
    RELEASE_REVIEW = "release:review"
    COMPUTE_READ = "compute:read"
    COMPUTE_WRITE = "compute:write"
    AUDIT_READ = "audit:read"
    USER_MANAGE = "user:manage"


@dataclass(frozen=True)
class PermissionTable:
    roles: dict[str, frozenset[str]]
    default_role: str

    def for_role(self, role: str) -> frozenset[str] | None:
        return self.roles.get(role)

    @classmethod
    def load(cls, path: Path) -> PermissionTable:
        raw = json.loads(path.read_text())
        roles = {name: frozenset(spec["permissions"]) for name, spec in raw["roles"].items()}
        return cls(roles=roles, default_role=raw["default_role"])
