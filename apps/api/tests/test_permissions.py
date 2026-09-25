"""The permission matrix (SPEC §4): every endpoint x every role.

A role without the endpoint's permission gets 403; a role with it gets past authorization (whatever the
business outcome: 200/201/204, or 404/409/422 for the made-up ids used here).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from skf_api.settings import REPO_ROOT

ROLES = json.loads((REPO_ROOT / "shared" / "permissions.json").read_text())["roles"]
MISSING = uuid.uuid4()

# (method, path, body, permissions of which any grants access)
ENDPOINTS: list[tuple[str, str, dict[str, Any] | None, tuple[str, ...]]] = [
    ("GET", "/api/v1/skills", None, ("skill:read",)),
    ("GET", "/api/v1/skills/g1-stairs", None, ("skill:read",)),
    ("POST", "/api/v1/skills/sync", None, ("skill:write",)),
    ("GET", "/api/v1/compute-targets", None, ("compute:read",)),
    ("GET", "/api/v1/compute-targets/schema/local_cpu", None, ("compute:write",)),
    (
        "POST",
        "/api/v1/compute-targets",
        {"name": "", "kind": "local_cpu", "steps_per_second": 1},
        ("compute:write",),
    ),
    ("PATCH", f"/api/v1/compute-targets/{MISSING}", {}, ("compute:write",)),
    ("PUT", f"/api/v1/compute-targets/{MISSING}/secret", {"secret": {}}, ("compute:write",)),
    ("POST", f"/api/v1/compute-targets/{MISSING}/check", None, ("compute:write",)),
    (
        "POST",
        "/api/v1/runs/estimate",
        {"skill_id": "nope", "compute_target_id": str(MISSING)},
        ("run:create_preset",),
    ),
    ("GET", "/api/v1/runs", None, ("run:read",)),
    ("POST", "/api/v1/runs", {"skill_id": "nope", "compute_target_id": str(MISSING)}, ("run:create_preset",)),
    ("GET", f"/api/v1/runs/{MISSING}", None, ("run:read",)),
    ("POST", f"/api/v1/runs/{MISSING}/cancel", None, ("run:read",)),
    ("POST", f"/api/v1/runs/{MISSING}/retry", None, ("run:read",)),
    ("POST", f"/api/v1/runs/{MISSING}/launch-decision", {"decision": "approve"}, ("run:approve_launch",)),
    ("POST", f"/api/v1/runs/{MISSING}/review", {"decision": "approve"}, ("release:review",)),
    ("POST", f"/api/v1/runs/{MISSING}/evaluate", {"checkpoint_id": str(MISSING)}, ("run:evaluate",)),
    ("GET", f"/api/v1/runs/{MISSING}/logs", None, ("run:read",)),
    ("GET", f"/api/v1/runs/{MISSING}/logs.txt", None, ("run:read",)),
    ("GET", f"/api/v1/runs/{MISSING}/metrics", None, ("run:read",)),
    ("GET", f"/api/v1/runs/{MISSING}/artifacts", None, ("run:read",)),
    ("GET", f"/api/v1/runs/{MISSING}/evaluations", None, ("run:read",)),
    ("GET", f"/api/v1/runs/{MISSING}/events", None, ("run:read",)),
    ("GET", f"/api/v1/artifacts/{MISSING}/url", None, ("run:read",)),
    ("GET", "/api/v1/approvals", None, ("release:review", "run:approve_launch")),
    ("GET", f"/api/v1/compare?run_ids={MISSING},{uuid.uuid4()}", None, ("run:read",)),
    ("GET", "/api/v1/dashboard", None, ("run:read",)),
    ("GET", "/api/v1/audit-events", None, ("audit:read",)),
    (
        "POST",
        "/api/v1/audit-events",
        {"action": "user.create", "entity_type": "user", "detail": {}},
        ("user:manage",),
    ),
]

CASES = [(method, path, body, perms, role) for method, path, body, perms in ENDPOINTS for role in ROLES]


def test_every_api_route_is_in_the_matrix() -> None:
    from skf_api.main import create_app

    def normalize(path: str) -> str:
        path = path.split("?")[0].replace(str(MISSING), "{id}")
        for name in ("{run_id}", "{target_id}", "{artifact_id}"):
            path = path.replace(name, "{id}")
        return path.replace("{skill_id}", "g1-stairs").replace("{kind}", "local_cpu")

    routes = {
        (method.upper(), normalize(path))
        for path, operations in create_app().openapi()["paths"].items()
        if path.startswith("/api/v1")
        for method in operations
    }
    documented = {(method, normalize(path)) for method, path, _, _ in ENDPOINTS}
    assert routes == documented | {("GET", "/api/v1/me")}


@pytest.mark.parametrize(
    ("method", "path", "body", "perms", "role"),
    CASES,
    ids=[f"{r}:{m} {p.split('?')[0]}" for m, p, _, _, r in CASES],
)
async def test_matrix(
    client: httpx.AsyncClient,
    auth: Callable[..., dict[str, str]],
    skills: list[str],
    method: str,
    path: str,
    body: dict[str, Any] | None,
    perms: tuple[str, ...],
    role: str,
) -> None:
    allowed = any(p in ROLES[role]["permissions"] for p in perms)
    response = await client.request(method, path, headers=auth(role), json=body)
    if allowed:
        assert response.status_code != 403, response.text
        assert response.status_code in (200, 201, 204, 404, 409, 422), response.text
    else:
        assert response.status_code == 403, response.text
        assert response.json()["error"]["code"] == "forbidden"


async def test_me_is_open_to_every_role(
    client: httpx.AsyncClient, auth: Callable[..., dict[str, str]]
) -> None:
    for role in ROLES:
        assert (await client.get("/api/v1/me", headers=auth(role))).status_code == 200


async def test_health_is_public(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok", "checks": {}}
    ready = await client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["checks"] == {"database": "ok", "redis": "ok"}
