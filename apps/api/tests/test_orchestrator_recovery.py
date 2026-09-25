"""Races and failures around the state machine (SPEC §6): cancels that overlap a submit or a status poll,
teardowns that fail, lost jobs, target release, work-dir cleanup and untrusted outputs."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
import sqlalchemy as sa
from api_fakes import FakeWorld, RecordingJobQueue
from test_state_machine import Auth, detail, launch, stage, step

from skf_api.backends.base import BackendError, JobSpec, JobState
from skf_api.context import AppContext
from skf_api.modules.compute.seeds import AWS_L40S, KAGGLE_T4
from skf_api.modules.runs.models import Stage, StageStatus
from skf_api.orchestrator import state_machine
from skf_api.orchestrator.reconciler import reconcile
from skf_api.orchestrator.state_machine import cancel_retry_marker

SSH_DOWN = BackendError("cannot reach 1.2.3.4 over SSH", retryable=True)


async def test_cancel_while_submitting_tears_down_the_new_job(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])

    async def cancel_meanwhile(spec: JobSpec) -> None:
        # The user cancels during a slow submit (box boot, rsync); the worker runs cancel_run concurrently.
        response = await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("ml_engineer"))
        assert response.status_code == 200
        await state_machine.cancel_run(ctx, uuid.UUID(run["id"]))
        waiting = stage(await detail(client, auth, run["id"]), "train")
        assert waiting["status"] == "provisioning"  # not "cancelled" while its job may still be starting
        assert waiting["message"] == "Cancelling once the job is submitted"

    world.during_submit = cancel_meanwhile
    await queue.drain(ctx)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "cancelled"
    assert [s["status"] for s in run["stages"]] == ["cancelled", "skipped", "skipped"]
    assert world.job("train").state is JobState.CANCELLED and world.cancelled == ["job-1"]
    assert str(targets[AWS_L40S]) in world.released


async def test_submission_for_a_stage_finished_meanwhile_is_abandoned(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])

    async def finish_elsewhere(spec: JobSpec) -> None:
        async with ctx.db.session() as session:
            await session.execute(
                sa.update(Stage).where(Stage.id == uuid.UUID(spec.stage_id)).values(status=StageStatus.FAILED)
            )
            await session.commit()

    world.during_submit = finish_elsewhere
    await queue.drain(ctx)
    train = stage(await detail(client, auth, run["id"]), "train")
    assert train["status"] == "failed" and train["external_url"] is None
    assert world.cancelled == ["job-1"] and str(targets[AWS_L40S]) in world.released


async def test_failed_teardown_keeps_the_stage_active_and_is_retried(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])
    run_id = uuid.UUID(run["id"])
    await queue.drain(ctx)
    world.cancel_errors = [SSH_DOWN]
    await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("ml_engineer"))
    await queue.drain(ctx)
    run = await detail(client, auth, run["id"])
    train = stage(run, "train")
    assert run["status"] == "running" and run["permissions"]["can_cancel"] is False
    assert train["status"] == "provisioning" and train["message"].startswith("Stopping the job failed")
    assert world.job("train").state is JobState.RUNNING

    await reconcile(ctx)  # still backing off
    assert ("cancel_run", run_id) not in queue.pending
    await ctx.redis.delete(cancel_retry_marker(run_id))  # the backoff expires
    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "cancelled" and stage(run, "train")["status"] == "cancelled"
    assert world.job("train").state is JobState.CANCELLED
    assert str(targets[AWS_L40S]) in world.released


async def test_teardown_gives_up_after_the_last_retry(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])
    run_id = uuid.UUID(run["id"])
    await queue.drain(ctx)
    world.cancel_errors = [SSH_DOWN] * (state_machine.MAX_CANCEL_RETRIES + 1)
    for _ in range(state_machine.MAX_CANCEL_RETRIES + 1):
        await state_machine.cancel_run(ctx, run_id)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "cancelled"
    assert stage(run, "train")["error"].startswith("The job could not be stopped")


async def test_submit_failure_releases_the_target(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    world.submit_error = BackendError("rsync to the instance failed")
    run = await launch(client, auth, targets[AWS_L40S])
    await queue.drain(ctx)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "failed" and stage(run, "train")["status"] == "failed"
    assert world.released == [str(targets[AWS_L40S])]  # submit may have booted the box


async def test_cancel_of_a_queued_stage_releases_the_target(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    world.submit_error = SSH_DOWN  # retryable: the stage goes back to queued and backs off
    run = await launch(client, auth, targets[AWS_L40S])
    await queue.drain(ctx)
    assert stage(await detail(client, auth, run["id"]), "train")["status"] == "queued"
    response = await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("ml_engineer"))
    assert queue.pending == [("cancel_run", uuid.UUID(run["id"]))]
    assert response.json()["status"] == "running"
    await queue.drain(ctx)
    assert (await detail(client, auth, run["id"]))["status"] == "cancelled"
    assert world.released == [str(targets[AWS_L40S])]


async def test_poll_does_not_overwrite_a_concurrent_cancel(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])
    await queue.drain(ctx)
    world.finish("train")

    async def cancel_meanwhile() -> None:
        world.during_status = None
        await client.post(f"/api/v1/runs/{run['id']}/cancel", headers=auth("ml_engineer"))
        await state_machine.cancel_run(ctx, uuid.UUID(run["id"]))

    world.during_status = cancel_meanwhile  # status() answers "succeeded", then the cancel lands
    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    assert run["status"] == "cancelled" and stage(run, "train")["status"] == "cancelled"
    artifacts = await client.get(f"/api/v1/runs/{run['id']}/artifacts", headers=auth("viewer"))
    assert artifacts.json() == []


async def test_failure_reported_by_a_stale_poll_keeps_the_cancel(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    run = await launch(client, auth, targets[AWS_L40S])
    await queue.drain(ctx)
    world.finish("train", JobState.FAILED, message="exit code 1")

    async def cancel_meanwhile() -> None:
        world.during_status = None
        await state_machine.cancel_run(ctx, uuid.UUID(run["id"]))

    world.during_status = cancel_meanwhile
    await step(ctx, queue)
    assert stage(await detail(client, auth, run["id"]), "train")["status"] == "cancelled"


async def test_lost_advance_run_is_recovered(
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
    await reconcile(ctx)
    assert ("advance_run", uuid.UUID(run["id"])) in queue.pending
    queue.pending.clear()  # the worker restarted before running it

    await step(ctx, queue)
    run = await detail(client, auth, run["id"])
    assert stage(run, "train")["status"] == "succeeded"
    assert stage(run, "evaluate")["status"] == "provisioning"


async def test_operator_retry_over_budget_needs_a_new_launch_approval(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
) -> None:
    await client.patch(
        f"/api/v1/compute-targets/{targets[KAGGLE_T4]}",
        headers=auth("admin"),
        json={"max_unapproved_gpu_hours": 0.1},
    )
    run = await launch(client, auth, targets[KAGGLE_T4], role="operator")
    approved = await client.post(
        f"/api/v1/runs/{run['id']}/launch-decision", headers=auth("ml_engineer"), json={"decision": "approve"}
    )
    assert approved.json()["status"] == "queued"
    await queue.drain(ctx)
    world.finish("train", JobState.FAILED, message="CUDA out of memory")
    await step(ctx, queue)

    retried = await client.post(f"/api/v1/runs/{run['id']}/retry", headers=auth("operator"))
    body = retried.json()
    assert body["status"] == "pending_approval" and body["launch_decided_by"] is None
    assert body["estimate"]["needs_approval"] is True
    assert queue.pending == [] and len(world.submitted) == 1

    reapproved = await client.post(
        f"/api/v1/runs/{run['id']}/launch-decision", headers=auth("ml_engineer"), json={"decision": "approve"}
    )
    assert reapproved.json()["status"] == "queued"
    await queue.drain(ctx)
    assert len(world.submitted) == 2


async def test_finished_stages_leave_no_work_dirs(
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
    assert (ctx.settings.work_dir / run["id"]).is_dir()  # the evaluate stage's inputs
    world.finish("evaluate")
    await step(ctx, queue)
    assert (await detail(client, auth, run["id"]))["status"] == "awaiting_review"
    assert not (ctx.settings.work_dir / run["id"]).exists()


async def test_symlinks_in_collected_outputs_are_not_published(
    client: httpx.AsyncClient,
    auth: Auth,
    ctx: AppContext,
    queue: RecordingJobQueue,
    world: FakeWorld,
    targets: dict[str, uuid.UUID],
    tmp_path: Path,
) -> None:
    secret = tmp_path / "worker.env"
    secret.write_text("DATABASE_URL=postgresql://secret\n")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "key.pem").write_text("key")

    def planted_links(out: Path) -> None:
        os.symlink(secret, out / "notes.txt")
        os.symlink(tmp_path / "private", out / "more")

    world.extra_outputs = planted_links
    run = await launch(client, auth, targets[KAGGLE_T4])
    await queue.drain(ctx)
    world.finish("train")
    await step(ctx, queue)
    assert stage(await detail(client, auth, run["id"]), "train")["status"] == "succeeded"
    artifacts = await client.get(f"/api/v1/runs/{run['id']}/artifacts", headers=auth("viewer"))
    names = {a["name"] for a in artifacts.json()}
    assert "params.pkl" in names and not names & {"notes.txt", "more/key.pem"}
