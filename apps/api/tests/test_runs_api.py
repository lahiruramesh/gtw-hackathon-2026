"""Run creation rules, listing, logs, metrics downsampling, compare, dashboard, audit."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import httpx
import sqlalchemy as sa
from api_fakes import FakeWorld, RecordingJobQueue

from skf_api.backends.base import JobState
from skf_api.context import AppContext
from skf_api.modules.compute.seeds import AWS_L40S, KAGGLE_T4
from skf_api.modules.ingest.service import IngestService, Line
from skf_api.modules.runs.metrics import downsample
from skf_api.modules.runs.models import Stage
from skf_api.orchestrator.reconciler import reconcile

Auth = Callable[..., dict[str, str]]


def body(target: uuid.UUID, **extra: Any) -> dict[str, Any]:
    return {
        "skill_id": "g1-stairs",
        "preset_id": "v11-finetune",
        "params": {},
        "compute_target_id": str(target),
        **extra,
    }


async def test_operator_rules(client: httpx.AsyncClient, auth: Auth, targets: dict[str, uuid.UUID]) -> None:
    ok = await client.post(
        "/api/v1/runs", headers=auth("operator"), json=body(targets[AWS_L40S], notes="first")
    )
    assert ok.status_code == 201
    assert (
        ok.json()["name"] == "g1-stairs-v11-finetune-1" and ok.json()["created_by"]["id"] == "user-operator"
    )
    same_values = await client.post(
        "/api/v1/runs", headers=auth("operator"), json=body(targets[AWS_L40S], params={"lr": 0.0001})
    )
    assert same_values.status_code == 201 and same_values.json()["name"] == "g1-stairs-v11-finetune-2"

    custom = await client.post(
        "/api/v1/runs", headers=auth("operator"), json=body(targets[AWS_L40S], params={"lr": 0.0002})
    )
    assert custom.status_code == 403
    no_preset = await client.post(
        "/api/v1/runs", headers=auth("operator"), json=body(targets[AWS_L40S], preset_id=None)
    )
    assert no_preset.status_code == 403
    assert (
        await client.post(
            "/api/v1/runs",
            headers=auth("ml_engineer"),
            json=body(targets[AWS_L40S], preset_id=None, params={"lr": 0.0002}),
        )
    ).status_code == 201


async def test_create_validation(
    client: httpx.AsyncClient, auth: Auth, ctx: AppContext, targets: dict[str, uuid.UUID]
) -> None:
    first = await client.post(
        "/api/v1/runs", headers=auth("admin"), json=body(targets[AWS_L40S], name="my-run")
    )
    assert first.status_code == 201
    duplicate = await client.post(
        "/api/v1/runs", headers=auth("admin"), json=body(targets[AWS_L40S], name="my-run")
    )
    assert duplicate.status_code == 409
    bad_name = await client.post(
        "/api/v1/runs", headers=auth("admin"), json=body(targets[AWS_L40S], name="My Run")
    )
    assert bad_name.status_code == 422
    unknown_preset = await client.post(
        "/api/v1/runs", headers=auth("admin"), json=body(targets[AWS_L40S], preset_id="v99")
    )
    assert unknown_preset.status_code == 422

    await client.patch(
        f"/api/v1/compute-targets/{targets[KAGGLE_T4]}", headers=auth("admin"), json={"enabled": False}
    )
    disabled = await client.post("/api/v1/runs", headers=auth("admin"), json=body(targets[KAGGLE_T4]))
    assert disabled.status_code == 422 and "disabled" in disabled.json()["error"]["message"]
    # The estimate names the same blocker, so the launch wizard can say so before anyone clicks Launch.
    estimate = await client.post(
        "/api/v1/runs/estimate", headers=auth("admin"), json=body(targets[KAGGLE_T4])
    )
    assert estimate.json()["blockers"] == [disabled.json()["error"]["message"]]


async def test_warm_start_from_checkpoint(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    parent = (
        await client.post("/api/v1/runs", headers=auth("ml_engineer"), json=body(targets[KAGGLE_T4]))
    ).json()
    await queue.drain(ctx)
    world.finish("train")
    await reconcile(ctx)
    ckpt = next(
        a
        for a in (await client.get(f"/api/v1/runs/{parent['id']}/artifacts", headers=auth("viewer"))).json()
        if a["step"] == 2000
    )

    denied = await client.post(
        "/api/v1/runs",
        headers=auth("operator"),
        json=body(targets[AWS_L40S], parent_checkpoint_id=ckpt["id"]),
    )
    assert denied.status_code == 403  # a warm start is a custom run
    child = await client.post(
        "/api/v1/runs",
        headers=auth("ml_engineer"),
        json=body(targets[AWS_L40S], parent_checkpoint_id=ckpt["id"]),
    )
    assert child.status_code == 201, child.text
    detail = child.json()
    assert (
        detail["parent"]["run"]["id"] == parent["id"] and detail["parent"]["checkpoint"]["id"] == ckpt["id"]
    )
    assert detail["params"]["init_from"] == ckpt["id"]

    queue.pending.clear()
    await queue.advance_run(uuid.UUID(detail["id"]))
    await queue.drain(ctx)
    spec = world.job("train").spec
    flag = spec.argv.index("--init-from")
    assert spec.argv[flag + 1] == "{input_dir}/parent/ckpt_00000002000.pkl"
    names = sorted(i.name for i in spec.inputs)
    assert names == ["parent/ckpt_00000002000.pkl", "parent/config.json", "parent/params.pkl"]
    assert all(i.source_ref is not None and i.source_ref.job_id == "job-1" for i in spec.inputs)
    parent_detail = (await client.get(f"/api/v1/runs/{parent['id']}", headers=auth("viewer"))).json()
    assert parent_detail["children"] == [{"id": detail["id"], "name": detail["name"], "status": "running"}]


async def test_list_filters_and_pagination(
    client: httpx.AsyncClient, auth: Auth, targets: dict[str, uuid.UUID]
) -> None:
    for role in ("operator", "ml_engineer", "ml_engineer"):
        await client.post("/api/v1/runs", headers=auth(role), json=body(targets[AWS_L40S]))
    page1 = (await client.get("/api/v1/runs?limit=2", headers=auth("viewer"))).json()
    page2 = (
        await client.get(f"/api/v1/runs?limit=2&cursor={page1['next_cursor']}", headers=auth("viewer"))
    ).json()
    names = [r["name"] for r in page1["items"] + page2["items"]]
    assert names == [f"g1-stairs-v11-finetune-{n}" for n in (3, 2, 1)] and page2["next_cursor"] is None
    mine = (await client.get("/api/v1/runs?created_by=me", headers=auth("operator"))).json()["items"]
    assert [r["name"] for r in mine] == ["g1-stairs-v11-finetune-1"]
    assert (await client.get("/api/v1/runs?q=finetune-2", headers=auth("viewer"))).json()["items"][0][
        "name"
    ] == "g1-stairs-v11-finetune-2"
    assert (await client.get("/api/v1/runs?status=failed", headers=auth("viewer"))).json()["items"] == []
    assert (await client.get("/api/v1/runs?cursor=garbage", headers=auth("viewer"))).status_code == 422


async def test_logs_paging_and_download(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    targets: dict[str, uuid.UUID],
) -> None:
    run = (
        await client.post("/api/v1/runs", headers=auth("ml_engineer"), json=body(targets[KAGGLE_T4]))
    ).json()
    await queue.drain(ctx)
    async with ctx.db.session() as session:
        stage = await session.scalar(
            sa.select(Stage).where(Stage.run_id == uuid.UUID(run["id"]), Stage.key == "train")
        )
        assert stage is not None
        await IngestService(ctx.redis, ctx.events, 1000).append_logs(
            session, stage, [Line(f"line {i}") for i in range(5)] + [Line("ERROR: disk full")]
        )

    url = f"/api/v1/runs/{run['id']}/logs"
    first = (await client.get(f"{url}?limit=4", headers=auth("viewer"))).json()
    assert [line["text"] for line in first["items"]] == ["line 0", "line 1", "line 2", "line 3"]
    rest = (await client.get(f"{url}?after_id={first['next_after_id']}", headers=auth("viewer"))).json()
    assert [line["text"] for line in rest["items"]] == ["line 4", "ERROR: disk full"] and rest[
        "next_after_id"
    ] is None
    errors = (await client.get(f"{url}?level=error", headers=auth("viewer"))).json()["items"]
    assert [line["text"] for line in errors] == ["ERROR: disk full"]
    assert len((await client.get(f"{url}?q=LINE%203", headers=auth("viewer"))).json()["items"]) == 1

    text = await client.get(f"{url}.txt", headers=auth("viewer"))
    assert text.status_code == 200 and text.headers["content-type"].startswith("text/plain")
    assert "attachment" in text.headers["content-disposition"]
    rows = text.text.splitlines()
    assert len(rows) == 6 and rows[0].endswith("[train] INFO  line 0") and "ERROR" in rows[5]


def test_downsample_keeps_ends() -> None:
    points = [(i, float(i)) for i in range(5000)]
    sampled = downsample(points)
    assert len(sampled) == 1000 and sampled[0] == (0, 0.0) and sampled[-1] == (4999, 4999.0)
    assert downsample(points[:10]) == points[:10]


async def test_dashboard_compare_and_audit(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    runs = [
        (await client.post("/api/v1/runs", headers=auth("ml_engineer"), json=body(targets[KAGGLE_T4]))).json()
        for _ in range(2)
    ]
    await queue.drain(ctx)
    world.finish("train", JobState.FAILED, message="boom")
    await reconcile(ctx)
    await queue.drain(ctx)

    dashboard = (await client.get("/api/v1/dashboard", headers=auth("viewer"))).json()
    assert dashboard["active_runs"] == 1 and dashboard["gate_pass_rate_30d"] is None
    assert [s["name"] for s in dashboard["skills"]] == [
        "Stair climbing (height scan)",
        "Step-length adjustment",
    ]
    assert next(s for s in dashboard["skills"] if s["id"] == "g1-stairs")["runs"] == 2
    assert [r["id"] for r in dashboard["recent_runs"]] == [runs[1]["id"], runs[0]["id"]]

    compare = await client.get(
        f"/api/v1/compare?run_ids={runs[0]['id']},{runs[1]['id']}", headers=auth("viewer")
    )
    assert compare.status_code == 200
    rows = compare.json()["headline_rows"]
    assert [r["label"] for r in rows] == ["Crossed", "Falls", "Certified height"]
    assert rows[0]["values"] == [None, None]
    one = await client.get(f"/api/v1/compare?run_ids={runs[0]['id']}", headers=auth("viewer"))
    assert one.status_code == 422

    created = await client.post(
        "/api/v1/audit-events",
        headers=auth("admin"),
        json={
            "action": "user.create",
            "entity_type": "user",
            "entity_id": "u-9",
            "detail": {"email": "new@skf.test", "password": "hunter2hunter2"},
        },
    )
    assert created.status_code == 201
    assert created.json()["detail"] == {"email": "new@skf.test"}  # secrets never stored
    events = (await client.get("/api/v1/audit-events?action=run.create", headers=auth("admin"))).json()
    assert len(events["items"]) == 2 and events["items"][0]["actor"]["id"] == "user-ml_engineer"
    page = (await client.get("/api/v1/audit-events?limit=1", headers=auth("admin"))).json()
    assert page["items"][0]["action"] == "user.create" and page["next_cursor"]


async def test_skills_catalog(client: httpx.AsyncClient, auth: Auth, skills: list[str]) -> None:
    listed = (await client.get("/api/v1/skills", headers=auth("viewer"))).json()
    assert [s["id"] for s in listed["items"]] == ["g1-stairs", "g1-stairs-bench", "g1-step-length", "ring-pick-drop"]
    detail = (await client.get("/api/v1/skills/g1-stairs", headers=auth("viewer"))).json()
    assert [p["id"] for p in detail["presets"]] == ["smoke", "v11-finetune"]
    assert detail["gate"][0]["label"] == "Crosses at least 90 % of held-out staircases"
    assert detail["params_schema"]["properties"]["scan_model"]["enum"] == ["uniform", "camera"]
    assert (await client.get("/api/v1/skills/nope", headers=auth("viewer"))).status_code == 404
    synced = await client.post("/api/v1/skills/sync", headers=auth("ml_engineer"))
    assert synced.json() == {"synced": ["g1-stairs", "g1-stairs-bench", "g1-step-length", "ring-pick-drop"], "errors": []}
