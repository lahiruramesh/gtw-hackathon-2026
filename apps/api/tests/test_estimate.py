"""Estimate and weekly-quota math (SPEC §4, §9.1)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import timedelta

import httpx
import pytest

from skf_api.context import AppContext
from skf_api.core.db import utcnow
from skf_api.modules.compute.seeds import AWS_L40S, KAGGLE_T4, LOCAL_CPU
from skf_api.modules.runs.models import Run, RunsOn, RunStatus, Stage, StageKind, StageStatus

Auth = Callable[..., dict[str, str]]


async def estimate(
    client: httpx.AsyncClient,
    auth: Auth,
    target: uuid.UUID,
    *,
    role: str = "operator",
    preset: str | None = "v11-finetune",
    params: dict[str, object] | None = None,
) -> dict:
    response = await client.post(
        "/api/v1/runs/estimate",
        headers=auth(role),
        json={
            "skill_id": "g1-stairs",
            "preset_id": preset,
            "params": params or {},
            "compute_target_id": str(target),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_gpu_target_estimate(
    client: httpx.AsyncClient, auth: Auth, targets: dict[str, uuid.UUID]
) -> None:
    # v11-finetune: 300M steps at 100k steps/s + 12 min overhead; strict test 25 min on the local CPU.
    body = await estimate(client, auth, targets[AWS_L40S])
    assert body["train_minutes"] == pytest.approx(62.0)
    assert body["total_minutes"] == pytest.approx(87.0)
    assert body["gpu_hours"] == pytest.approx(62 / 60, abs=1e-3)
    assert body["cost"] == pytest.approx(62 / 60 * 2.25, abs=0.01)
    assert body["needs_approval"] is False and body["warnings"] == [] and body["blockers"] == []


async def test_operator_over_budget_needs_approval(
    client: httpx.AsyncClient, auth: Auth, targets: dict[str, uuid.UUID]
) -> None:
    # Kaggle T4: 300M / 36k steps/s = 138.9 min + 8 min = 2.45 GPU-h; operators may launch up to 4 unapproved.
    assert (await estimate(client, auth, targets[KAGGLE_T4]))["needs_approval"] is False
    await client.patch(
        f"/api/v1/compute-targets/{targets[KAGGLE_T4]}",
        headers=auth("admin"),
        json={"max_unapproved_gpu_hours": 2},
    )
    body = await estimate(client, auth, targets[KAGGLE_T4])
    assert body["needs_approval"] is True
    assert "2.4 GPU-h exceeds the 2 GPU-h" in body["reasons"][0]
    assert (await estimate(client, auth, targets[KAGGLE_T4], role="ml_engineer"))["needs_approval"] is False


async def test_weekly_quota(
    client: httpx.AsyncClient, auth: Auth, ctx: AppContext, targets: dict[str, uuid.UUID]
) -> None:
    kaggle = targets[KAGGLE_T4]
    async with ctx.db.session() as session:
        run = Run(
            name="quota-probe",
            skill_id="g1-stairs",
            params={},
            status=RunStatus.FAILED,
            compute_target_id=kaggle,
            created_by_id="u",
            created_by_name="u",
            created_by_role="admin",
        )
        session.add(run)
        await session.flush()
        now = utcnow()
        for key, hours, finished in (
            ("a", 20.0, now - timedelta(days=1)),
            ("b", 5.0, now - timedelta(days=8)),
            ("c", 8.0, None),
        ):
            session.add(
                Stage(
                    run_id=run.id,
                    key=key,
                    kind=StageKind.TRAIN,
                    title=key,
                    position=0,
                    runs_on=RunsOn.TARGET,
                    compute_target_id=kaggle,
                    gpu_seconds=hours * 3600,
                    status=StageStatus.RUNNING if finished is None else StageStatus.SUCCEEDED,
                    finished_at=finished,
                )
            )
        await session.commit()

    targets_view = (await client.get("/api/v1/compute-targets", headers=auth("viewer"))).json()
    usage = next(t for t in targets_view if t["id"] == str(kaggle))["usage"]
    # 20 h finished yesterday + 8 h still running count; 5 h finished 8 days ago does not.
    assert usage == {
        "gpu_hours_7d": 28.0,
        "cost_7d": 0.0,
        "active_stages": 1,
        "quota_left_hours": 2.0,
        "quota_source": "ledger",
    }

    body = await estimate(client, auth, kaggle)
    assert any("only 2.0 GPU-h of the weekly quota is left" in w for w in body["warnings"])
    dashboard = (await client.get("/api/v1/dashboard", headers=auth("viewer"))).json()
    assert next(t for t in dashboard["targets"] if t["name"] == KAGGLE_T4)["quota_left_hours"] == 2.0


async def test_local_cpu_rules(client: httpx.AsyncClient, auth: Auth, targets: dict[str, uuid.UUID]) -> None:
    smoke = await estimate(client, auth, targets[LOCAL_CPU], preset="smoke")
    assert smoke["gpu_hours"] == 0 and smoke["warnings"] == [] and smoke["blockers"] == []
    full = await estimate(client, auth, targets[LOCAL_CPU], role="ml_engineer", preset=None)
    assert (
        full["blockers"] == ["Only smoke runs may train on the local CPU target"] and full["warnings"] == []
    )
    response = await client.post(
        "/api/v1/runs",
        headers=auth("ml_engineer"),
        json={"skill_id": "g1-stairs", "params": {}, "compute_target_id": str(targets[LOCAL_CPU])},
    )
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Only smoke runs may train on the local CPU target"


async def test_estimate_validates_params(
    client: httpx.AsyncClient, auth: Auth, targets: dict[str, uuid.UUID]
) -> None:
    response = await client.post(
        "/api/v1/runs/estimate",
        headers=auth("operator"),
        json={
            "skill_id": "g1-stairs",
            "params": {"timesteps": 5, "bogus": 1},
            "compute_target_id": str(targets[AWS_L40S]),
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["details"]["params"] == {
        "bogus": "unknown parameter",
        "timesteps": "must be >= 20000",
    }
