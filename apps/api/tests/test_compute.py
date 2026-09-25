"""Compute targets: CRUD, write-only secrets validated against the backend model, health checks."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import httpx
import sqlalchemy as sa
from api_fakes import FakeWorld

from skf_api.backends.base import HealthReport, HealthStatus
from skf_api.context import AppContext
from skf_api.modules.audit.models import AuditEvent
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.compute.seeds import AWS_L40S, KAGGLE_T4, default_seeds, seed_targets

Auth = Callable[..., dict[str, str]]

KAGGLE = {
    "name": "Kaggle (team)",
    "kind": "kaggle",
    "steps_per_second": 36000,
    "overhead_minutes": 8,
    "weekly_quota_gpu_hours": 30,
    "config": {"username": "skf-team"},
}


async def test_create_update_and_list(client: httpx.AsyncClient, auth: Auth, skills: list[str]) -> None:
    created = await client.post("/api/v1/compute-targets", headers=auth("admin"), json=KAGGLE)
    assert created.status_code == 201, created.text
    target = created.json()
    assert target["enabled"] is False and target["has_secret"] is False
    assert target["health"] == {"status": "unknown", "message": "Not checked yet", "checked_at": None}
    assert target["usage"] == {
        "gpu_hours_7d": 0.0,
        "cost_7d": 0.0,
        "active_stages": 0,
        "quota_left_hours": 30.0,
        "quota_source": "ledger",
    }

    duplicate = await client.post("/api/v1/compute-targets", headers=auth("admin"), json=KAGGLE)
    assert duplicate.status_code == 409
    bad_config = await client.post(
        "/api/v1/compute-targets",
        headers=auth("admin"),
        json={**KAGGLE, "name": "other", "config": {"username": "no spaces allowed"}},
    )
    assert bad_config.status_code == 422 and "username" in bad_config.json()["error"]["message"]

    patched = await client.patch(
        f"/api/v1/compute-targets/{target['id']}",
        headers=auth("admin"),
        json={"description": "Team account", "max_concurrent": 3},
    )
    assert patched.json()["description"] == "Team account" and patched.json()["max_concurrent"] == 3
    null_field = await client.patch(
        f"/api/v1/compute-targets/{target['id']}", headers=auth("admin"), json={"steps_per_second": None}
    )
    assert null_field.status_code == 422

    listed = (await client.get("/api/v1/compute-targets", headers=auth("viewer"))).json()
    assert [t["name"] for t in listed] == ["Kaggle (team)"]
    assert "secret" not in listed[0] and "secret_enc" not in listed[0]


async def test_secret_is_write_only_and_validated(
    client: httpx.AsyncClient, auth: Auth, ctx: AppContext, skills: list[str]
) -> None:
    target = (await client.post("/api/v1/compute-targets", headers=auth("admin"), json=KAGGLE)).json()
    enable_early = await client.patch(
        f"/api/v1/compute-targets/{target['id']}", headers=auth("admin"), json={"enabled": True}
    )
    assert enable_early.status_code == 422 and "credentials" in enable_early.json()["error"]["message"]

    wrong = await client.put(
        f"/api/v1/compute-targets/{target['id']}/secret",
        headers=auth("admin"),
        json={"secret": {"api_key": "sekrit-value"}},
    )
    assert wrong.status_code == 422
    assert "sekrit-value" not in wrong.text  # never echoed back

    response = await client.put(
        f"/api/v1/compute-targets/{target['id']}/secret",
        headers=auth("admin"),
        json={"secret": {"key": "sekrit-value"}},
    )
    assert response.status_code == 204
    enabled = await client.patch(
        f"/api/v1/compute-targets/{target['id']}", headers=auth("admin"), json={"enabled": True}
    )
    assert enabled.json()["enabled"] is True and enabled.json()["has_secret"] is True
    assert "sekrit-value" not in enabled.text

    async with ctx.db.session() as session:
        row = await session.get(ComputeTarget, uuid.UUID(target["id"]))
        assert row is not None and row.secret_enc is not None
        assert b"sekrit" not in row.secret_enc
        assert ctx.secrets.decrypt_json(row.secret_enc) == {"key": "sekrit-value"}
        audit = (
            await session.scalars(sa.select(AuditEvent).where(AuditEvent.action == "target.secret_set"))
        ).one()
    assert audit.detail == {"fields": ["key"]} and audit.actor_id == "user-admin"


async def test_health_check(
    client: httpx.AsyncClient, auth: Auth, world: FakeWorld, targets: dict[str, uuid.UUID]
) -> None:
    before = await client.get("/api/v1/compute-targets", headers=auth("admin"))
    usage = next(t for t in before.json() if t["id"] == str(targets[KAGGLE_T4]))["usage"]
    assert (usage["quota_left_hours"], usage["quota_source"]) == (30, "ledger")

    # Kaggle reports its own remaining quota, which counts usage outside the studio.
    details = {"gpu_used_hours": 29.2, "gpu_remaining_hours": 0.8, "gpu_total_hours": 30.0}
    world.health = HealthReport(HealthStatus.DEGRADED, "0.8 GPU-hours of quota left", details)
    checked = await client.post(f"/api/v1/compute-targets/{targets[KAGGLE_T4]}/check", headers=auth("admin"))
    assert checked.status_code == 200
    health = checked.json()["health"]
    assert health["status"] == "degraded" and health["message"] == "0.8 GPU-hours of quota left"
    assert health["checked_at"] is not None
    usage = checked.json()["usage"]
    assert (usage["quota_left_hours"], usage["quota_source"]) == (0.8, "provider")

    estimate = await client.post(
        "/api/v1/runs/estimate",
        headers=auth("operator"),
        json={
            "skill_id": "g1-stairs",
            "preset_id": "v11-finetune",
            "compute_target_id": str(targets[KAGGLE_T4]),
        },
    )
    assert any("weekly quota" in w for w in estimate.json()["warnings"])


async def test_backend_schema(client: httpx.AsyncClient, auth: Auth, skills: list[str]) -> None:
    body = (await client.get("/api/v1/compute-targets/schema/aws_ec2", headers=auth("admin"))).json()
    assert "instance_name" in body["config"]["properties"]
    assert "ssh_private_key" in body["secret"]["properties"]
    assert (await client.get("/api/v1/compute-targets/schema/gcp", headers=auth("admin"))).status_code == 422


async def test_seeds_are_idempotent_and_valid(ctx: AppContext, skills: list[str]) -> None:
    from skf_api.modules.compute.service import _validate

    async with ctx.db.session() as session:
        assert await seed_targets(session, default_seeds("skf-team")) == ["Local CPU", KAGGLE_T4, AWS_L40S]
        assert await seed_targets(session, default_seeds("skf-team")) == []
        rows = (await session.scalars(sa.select(ComputeTarget).order_by(ComputeTarget.name))).all()
    by_name = {t.name: t for t in rows}
    assert by_name["Local CPU"].enabled and not by_name[KAGGLE_T4].enabled and not by_name[AWS_L40S].enabled
    assert by_name[AWS_L40S].config["instance_type"] == "g6e.xlarge"
    assert "aws_profile" not in by_name[AWS_L40S].config  # the worker uses the host's IAM role
    assert by_name[KAGGLE_T4].weekly_quota_gpu_hours == 30
    for target in rows:
        _validate(ctx.backends.schema(target.kind)["config"], target.config, "config")
