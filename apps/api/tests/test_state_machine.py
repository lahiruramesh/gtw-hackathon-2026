"""The run/stage state machine end to end (SPEC §6), driven through the API with fake backends."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from api_fakes import FakeWorld, RecordingJobQueue

from skf_api.backends.base import JobState
from skf_api.context import AppContext
from skf_api.modules.compute.seeds import AWS_L40S, KAGGLE_T4, LOCAL_CPU
from skf_api.modules.runs.models import Stage, StageStatus
from skf_api.orchestrator.reconciler import reconcile

Auth = Callable[..., dict[str, str]]


async def launch(
    client: httpx.AsyncClient,
    auth: Auth,
    target_id: uuid.UUID,
    *,
    role: str = "ml_engineer",
    preset: str = "v11-finetune",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/runs",
        headers=auth(role),
        json={
            "skill_id": "g1-stairs",
            "preset_id": preset,
            "params": params or {},
            "compute_target_id": str(target_id),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def detail(client: httpx.AsyncClient, auth: Auth, run_id: str, role: str = "viewer") -> dict[str, Any]:
    response = await client.get(f"/api/v1/runs/{run_id}", headers=auth(role))
    assert response.status_code == 200, response.text
    return response.json()


def stage(run: dict[str, Any], key: str) -> dict[str, Any]:
    return next(s for s in run["stages"] if s["key"] == key)


async def step(ctx: AppContext, queue: RecordingJobQueue) -> None:
    """One orchestrator tick: run queued jobs, reconcile, run what that enqueued."""
    await queue.drain(ctx)
    await reconcile(ctx)
    await queue.drain(ctx)


async def test_happy_path_to_approved_release(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])
    assert run["status"] == "queued"
    assert [s["status"] for s in run["stages"]] == ["pending", "pending", "pending"]

    await queue.drain(ctx)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "running"
    train = stage(run, "train")
    assert train["status"] == "provisioning" and train["external_url"].startswith("https://jobs.test/")
    spec = world.job("train").spec
    assert spec.argv[:3] == ["python", "-m", "g1pipe.train"]
    assert "--lr" in spec.argv and spec.ingest is None  # AWS without PUBLIC_INGEST_URL: logs are polled

    world.job("train").logs = ["step 0", "Traceback (most recent call last):", "warning: slow"]
    world.job("train").gpu_seconds = 1800
    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    train = stage(run, "train")
    assert train["status"] == "running"
    assert train["gpu_seconds"] == 1800 and train["cost"] == pytest.approx(1.125)
    logs = (await client.get(f"/api/v1/runs/{run['id']}/logs", headers=auth("viewer"))).json()["items"]
    assert [line["level"] for line in logs] == ["info", "error", "warn"]

    world.finish("train", gpu_seconds=3600)
    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    assert stage(run, "train")["status"] == "succeeded"
    assert stage(run, "evaluate")["status"] == "provisioning"
    evaluate_spec = world.job("evaluate").spec
    assert evaluate_spec.ingest is not None and evaluate_spec.ingest.url == "http://api.test/ingest/v1"
    assert sorted(i.name for i in evaluate_spec.inputs) == ["train/config.json", "train/params.pkl"]
    assert str(targets[AWS_L40S]) in world.released  # the GPU box stops once nothing needs it

    artifacts = (await client.get(f"/api/v1/runs/{run['id']}/artifacts", headers=auth("viewer"))).json()
    by_name = {a["name"]: a for a in artifacts}
    assert (
        by_name["ckpt_00000002000.pkl"]["kind"] == "checkpoint"
        and by_name["ckpt_00000002000.pkl"]["step"] == 2000
    )
    assert by_name["params.pkl"]["kind"] == "params"
    metrics = (await client.get(f"/api/v1/runs/{run['id']}/metrics", headers=auth("viewer"))).json()
    assert metrics["keys"] == ["eval/episode_crossed", "eval/episode_reward"]  # std / per-term keys dropped

    world.finish("evaluate")
    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "awaiting_review"
    assert run["gate"]["verdict"] == "pass" and run["gate"]["review_status"] == "pending"
    assert stage(run, "gate")["status"] == "succeeded"
    assert run["headline"][0] == {"label": "Crossed", "value": 100.0, "unit": "%"}
    assert run["gpu_hours"] == 1.0 and run["cost"] == 2.25

    own = await client.post(
        f"/api/v1/runs/{run['id']}/review",
        headers=auth("admin", sub="user-ml_engineer"),
        json={"decision": "approve"},
    )
    assert own.status_code == 403  # no approving your own release
    reviewed = await client.post(
        f"/api/v1/runs/{run['id']}/review",
        headers=auth("safety_reviewer"),
        json={"decision": "approve", "comment": "ok for gantry"},
    )
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["status"] == "approved"
    assert reviewed.json()["gate"]["reviewer"]["id"] == "user-safety_reviewer"


async def test_gate_failure(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    world.strict_pass = False
    run = await launch(client, auth, targets[KAGGLE_T4])
    await queue.drain(ctx)
    world.finish("train")
    await step(ctx, queue)
    world.finish("evaluate")
    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "gate_failed"
    assert run["gate"]["review_status"] == "not_required"
    assert [c["passed"] for c in run["gate"]["criteria"]] == [False, False, False]
    review = await client.post(
        f"/api/v1/runs/{run['id']}/review", headers=auth("safety_reviewer"), json={"decision": "approve"}
    )
    assert review.status_code == 409


async def test_failure_then_retry(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[KAGGLE_T4], role="operator")
    await queue.drain(ctx)
    world.finish("train", JobState.FAILED, message="CUDA out of memory")
    await step(ctx, queue)
    run = await detail(client, auth, run["id"], role="operator")
    assert run["status"] == "failed" and run["error"] == "CUDA out of memory"
    assert [s["status"] for s in run["stages"]] == ["failed", "skipped", "skipped"]
    assert run["permissions"]["can_retry"] is True

    other = await client.post(f"/api/v1/runs/{run['id']}/retry", headers=auth("operator", sub="someone-else"))
    assert other.status_code == 403
    retried = await client.post(f"/api/v1/runs/{run['id']}/retry", headers=auth("operator"))
    assert retried.status_code == 200, retried.text
    body = retried.json()
    assert body["status"] == "queued"
    assert stage(body, "train")["attempt"] == 2 and stage(body, "evaluate")["status"] == "pending"

    await queue.drain(ctx)
    world.finish("train")
    await step(ctx, queue)
    world.finish("evaluate")
    await step(ctx, queue)
    assert (await detail(client, auth, run["id"]))["status"] == "awaiting_review"
    async with ctx.db.session() as session:
        attempts = (
            await session.scalars(
                sa.select(Stage.attempt)
                .where(Stage.run_id == uuid.UUID(run["id"]), Stage.key == "train")
                .order_by(Stage.attempt)
            )
        ).all()
    assert list(attempts) == [1, 2]


async def test_cancel_tears_down_remote_job(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])
    await queue.drain(ctx)
    denied = await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("operator"))
    assert denied.status_code == 403  # not the operator's run
    response = await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("ml_engineer"))
    assert response.status_code == 200
    assert response.json()["permissions"]["can_cancel"] is False
    assert queue.pending == [("cancel_run", uuid.UUID(run["id"]))]
    await queue.drain(ctx)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "cancelled"
    assert [s["status"] for s in run["stages"]] == ["cancelled", "skipped", "skipped"]
    assert world.cancelled == ["job-1"]
    again = await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("ml_engineer"))
    assert again.status_code == 409


async def test_cancel_before_start_needs_no_worker(
    client: httpx.AsyncClient, auth: Auth, queue: RecordingJobQueue, targets: dict[str, uuid.UUID]
) -> None:
    run = await launch(client, auth, targets[KAGGLE_T4])
    response = await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("ml_engineer"))
    assert response.json()["status"] == "cancelled"
    assert ("cancel_run", uuid.UUID(run["id"])) not in queue.pending


async def test_launch_approval(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    targets: dict[str, uuid.UUID],
) -> None:
    # Operators need approval above the target's max_unapproved_gpu_hours (Kaggle: 4 GPU-h).
    await client.patch(
        f"/api/v1/compute-targets/{targets[KAGGLE_T4]}",
        headers=auth("admin"),
        json={"max_unapproved_gpu_hours": 1},
    )
    run = await launch(client, auth, targets[KAGGLE_T4], role="operator")
    assert run["status"] == "pending_approval"
    assert run["estimate"]["needs_approval"] is True and run["estimate"]["reasons"]
    assert queue.pending == []

    queues = (await client.get("/api/v1/approvals", headers=auth("ml_engineer"))).json()
    assert [r["id"] for r in queues["launches"]] == [run["id"]] and queues["releases"] == []
    self_approve = await client.post(
        f"/api/v1/runs/{run['id']}/launch-decision",
        headers=auth("ml_engineer", sub="user-operator"),
        json={"decision": "approve"},
    )
    assert self_approve.status_code == 403
    approved = await client.post(
        f"/api/v1/runs/{run['id']}/launch-decision", headers=auth("ml_engineer"), json={"decision": "approve"}
    )
    assert approved.json()["status"] == "queued"
    assert approved.json()["launch_decided_by"]["id"] == "user-ml_engineer"
    await queue.drain(ctx)
    assert stage(await detail(client, auth, run["id"]), "train")["status"] == "provisioning"

    # The same launch by an ml_engineer is queued directly.
    direct = await launch(client, auth, targets[KAGGLE_T4])
    assert direct["status"] == "queued"


async def test_launch_rejection(client: httpx.AsyncClient, auth: Auth, targets: dict[str, uuid.UUID]) -> None:
    await client.patch(
        f"/api/v1/compute-targets/{targets[KAGGLE_T4]}",
        headers=auth("admin"),
        json={"max_unapproved_gpu_hours": 1},
    )
    run = await launch(client, auth, targets[KAGGLE_T4], role="operator")
    rejected = await client.post(
        f"/api/v1/runs/{run['id']}/launch-decision",
        headers=auth("admin"),
        json={"decision": "reject", "comment": "use the smoke preset first"},
    )
    body = rejected.json()
    assert body["status"] == "cancelled"
    assert body["error"] == "launch rejected: use the smoke preset first"


async def test_max_concurrent_keeps_stage_queued(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    targets: dict[str, uuid.UUID],
) -> None:
    first = await launch(client, auth, targets[AWS_L40S])  # AWS L40S: max_concurrent 1
    second = await launch(client, auth, targets[AWS_L40S])
    await queue.drain(ctx)
    assert stage(await detail(client, auth, first["id"]), "train")["status"] == "provisioning"
    waiting = stage(await detail(client, auth, second["id"]), "train")
    assert waiting["status"] == "queued" and waiting["message"] == "Waiting for a free slot on AWS L40S"


async def test_worker_restart_mid_submit_is_requeued(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[KAGGLE_T4])
    await queue.drain(ctx)
    async with ctx.db.session() as session:  # a claim that never recorded its external_ref
        await session.execute(
            sa.update(Stage)
            .where(Stage.run_id == uuid.UUID(run["id"]), Stage.key == "train")
            .values(external_ref=None)
        )
        await session.commit()
    await reconcile(ctx)
    train = stage(await detail(client, auth, run["id"]), "train")
    assert train["status"] == "queued" and "Re-queued" in train["message"]
    await queue.drain(ctx)
    assert stage(await detail(client, auth, run["id"]), "train")["status"] == "provisioning"
    assert len(world.jobs) == 2


async def test_advance_run_is_idempotent(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[KAGGLE_T4])
    run_id = uuid.UUID(run["id"])
    await queue.advance_run(run_id)
    await queue.advance_run(run_id)
    ran = await queue.drain(ctx)
    assert ran.count("execute_stage") == 1
    async with ctx.db.session() as session:
        statuses = (
            await session.scalars(
                sa.select(Stage.status).where(Stage.run_id == run_id).order_by(Stage.position)
            )
        ).all()
    assert list(statuses) == [StageStatus.PROVISIONING, StageStatus.PENDING, StageStatus.PENDING]


async def test_evaluate_checkpoint_side_stage(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[KAGGLE_T4])
    await queue.drain(ctx)
    world.finish("train")
    await step(ctx, queue)
    world.finish("evaluate")
    await step(ctx, queue)
    artifacts = (await client.get(f"/api/v1/runs/{run['id']}/artifacts", headers=auth("viewer"))).json()
    ckpt = next(a for a in artifacts if a["step"] == 1000)

    denied = await client.post(
        f"/api/v1/runs/{run['id']}/evaluate", headers=auth("operator"), json={"checkpoint_id": ckpt["id"]}
    )
    assert denied.status_code == 403
    response = await client.post(
        f"/api/v1/runs/{run['id']}/evaluate", headers=auth("ml_engineer"), json={"checkpoint_id": ckpt["id"]}
    )
    assert response.status_code == 200, response.text
    side = stage(response.json(), "evaluate-ckpt-1000")
    assert side["status"] == "queued" and side["checkpoint"]["id"] == ckpt["id"]

    await queue.drain(ctx)
    spec = world.job("evaluate-ckpt-1000").spec
    assert "{input_dir}/train/ckpt_00000001000.pkl" in spec.argv
    assert sorted(i.name for i in spec.inputs) == [
        "train/ckpt_00000001000.pkl",
        "train/config.json",
        "train/params.pkl",
    ]
    world.finish("evaluate-ckpt-1000")
    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "awaiting_review"  # a side stage never moves the run
    evaluations = (await client.get(f"/api/v1/runs/{run['id']}/evaluations", headers=auth("viewer"))).json()
    assert {e["stage_key"] for e in evaluations} == {"evaluate", "evaluate-ckpt-1000"}
    assert (
        next(e for e in evaluations if e["stage_key"] == "evaluate-ckpt-1000")["checkpoint"]["step"] == 1000
    )


async def test_local_smoke_run_phones_home(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[LOCAL_CPU], role="operator", preset="smoke")
    assert run["status"] == "queued" and run["estimate"]["gpu_hours"] == 0
    await queue.drain(ctx)
    spec = world.job("train").spec
    assert spec.ingest is not None and spec.ingest.url == "http://api.test/ingest/v1"
    assert "--smoke" in spec.argv
    ingest = await client.post(
        f"/ingest/v1/stages/{spec.stage_id}/heartbeat",
        headers={"Authorization": f"Bearer {spec.ingest.token}"},
        json={"progress": 0.5},
    )
    assert ingest.status_code == 200
    train = stage(await detail(client, auth, run["id"]), "train")
    assert train["status"] == "running" and train["progress"] == 0.5
