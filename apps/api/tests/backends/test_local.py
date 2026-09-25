"""local_cpu runs real (tiny) processes through the real reporter."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from backend_fakes import FakeRunner, result

from skf_api.backends import local
from skf_api.backends.base import (
    ExternalRef,
    HealthStatus,
    InputFile,
    JobSpec,
    JobState,
    JobStatus,
    TargetContext,
)
from skf_api.backends.local import LocalCpuBackend

JOB = """
import pathlib, shutil, sys
out = pathlib.Path(sys.argv[1]); inp = pathlib.Path(sys.argv[2])
shutil.copy(inp / "train" / "params.pkl", out / "params.pkl")
(out / "progress.csv").write_text("step,wall_s,eval/episode_reward\\n0,1.0,-3.5\\n")
print("hello from the job")
print("MuJoCo Warp: iterations limit reached")
print("secret:", __import__("os").environ.get("SKF_TEST_SECRET"))
print("no newline at the end", end="")
"""


def job_spec(
    tmp_path: Path,
    code: str,
    *,
    stage_id: str = "stage-0001-abcd",
    argv: list[str] | None = None,
    inputs: list[InputFile] | None = None,
) -> JobSpec:
    return JobSpec(
        run_id="run-1",
        run_name="g1-test",
        stage_id=stage_id,
        stage_key="train",
        kind="train",
        argv=argv or ["python", "-c", code, "{out_dir}", "{input_dir}"],
        inputs=inputs or [],
        timeout_minutes=5,
    )


def backend(make_ctx: Callable[..., TargetContext], **kwargs: object) -> LocalCpuBackend:
    return LocalCpuBackend(make_ctx({"python": sys.executable}), kill_grace_seconds=2, **kwargs)  # type: ignore[arg-type]


def _job_dir(ref: ExternalRef) -> Path:
    return Path(ref.data["job_dir"])


async def wait_terminal(b: LocalCpuBackend, ref: ExternalRef, limit: float = 20) -> JobStatus:
    for _ in range(int(limit / 0.1)):
        status = await b.status(ref)
        if status.state.terminal:
            return status
        await asyncio.sleep(0.1)
    raise AssertionError("job did not finish")


async def test_runs_a_job_end_to_end(
    make_ctx: Callable[..., TargetContext], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SKF_TEST_SECRET", "worker-only")
    params = tmp_path / "params.pkl"
    params.write_bytes(b"weights")
    b = backend(make_ctx)
    ref = await b.submit(job_spec(tmp_path, JOB, inputs=[InputFile("train/params.pkl", params)]))

    status = await wait_terminal(b, ref)
    assert status.state is JobState.SUCCEEDED

    batch = await b.fetch_logs(ref, None)
    assert "hello from the job" in batch.lines
    assert "secret: None" in batch.lines, "worker environment must not leak into the job"
    assert "no newline at the end" in batch.lines
    assert not any("iterations limit" in line for line in batch.lines)
    assert any("1 noise lines dropped" in line for line in batch.lines)
    assert (await b.fetch_logs(ref, batch.cursor)).lines == []

    dest = tmp_path / "collected"
    await b.collect(ref, dest)
    assert (dest / "params.pkl").read_bytes() == b"weights"
    assert (dest / "progress.csv").is_file()
    assert "hello from the job" in (dest / "job.log").read_text()


async def test_release_removes_finished_job_dirs_only(
    make_ctx: Callable[..., TargetContext], tmp_path: Path
) -> None:
    b = backend(make_ctx)
    finished = await b.submit(job_spec(tmp_path, "print('done')", stage_id="stage-done-0001"))
    await wait_terminal(b, finished)
    running = await b.submit(job_spec(tmp_path, "import time; time.sleep(30)", stage_id="stage-live-0002"))
    finished_dir, running_dir = _job_dir(finished), _job_dir(running)
    try:
        await b.release()
        assert not finished_dir.exists()
        assert running_dir.is_dir()
    finally:
        await b.cancel(running)
    await b.release()
    assert not running_dir.exists()
    assert not running_dir.parent.exists()  # the run's dir, once empty


async def test_failed_job_reports_exit_code(make_ctx: Callable[..., TargetContext], tmp_path: Path) -> None:
    b = backend(make_ctx)
    ref = await b.submit(job_spec(tmp_path, "import sys; print('boom'); sys.exit(3)"))
    status = await wait_terminal(b, ref)
    assert (status.state, status.message) == (JobState.FAILED, "exit code 3")


async def test_submit_is_idempotent(make_ctx: Callable[..., TargetContext], tmp_path: Path) -> None:
    b = backend(make_ctx)
    spec = job_spec(tmp_path, "print('once')")
    first = await b.submit(spec)
    second = await b.submit(spec)
    assert first == second
    await wait_terminal(b, first)
    assert (await b.fetch_logs(first, None)).lines.count("once") == 1


async def test_cancel_kills_the_whole_process_group(
    make_ctx: Callable[..., TargetContext], tmp_path: Path
) -> None:
    code = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "open(sys.argv[1] + '/child.pid', 'w').write(str(child.pid)); print('ready', flush=True); "
        "time.sleep(60)"
    )
    b = backend(make_ctx)
    ref = await b.submit(job_spec(tmp_path, code))
    child_pid_file = Path(ref.data["job_dir"]) / "out" / "child.pid"
    for _ in range(100):
        if child_pid_file.exists() and child_pid_file.read_text():
            break
        await asyncio.sleep(0.1)
    grandchild = int(child_pid_file.read_text())

    await b.cancel(ref)
    assert (await b.status(ref)).state is JobState.CANCELLED
    await b.cancel(ref)  # idempotent
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("grandchild survived cancel")


async def test_status_after_worker_restart_uses_pid_and_exit_file(
    make_ctx: Callable[..., TargetContext], tmp_path: Path
) -> None:
    b = backend(make_ctx)
    ref = await b.submit(job_spec(tmp_path, "import time; time.sleep(1.5)"))
    local._children.pop(int(ref.data["pid"]))  # as if another worker process had started it
    restored = ExternalRef.from_json(ref.to_json())
    fresh = backend(make_ctx)
    assert (await fresh.status(restored)).state is JobState.RUNNING
    status = await wait_terminal(fresh, restored)
    assert status.state is JobState.SUCCEEDED


async def test_unrelated_process_with_the_same_pid_is_not_our_job(
    make_ctx: Callable[..., TargetContext], tmp_path: Path
) -> None:
    job_dir = tmp_path / "gone"
    job_dir.mkdir()
    ref = ExternalRef(
        local.BackendKind.LOCAL_CPU,
        str(os.getpid()),
        {"pid": os.getpid(), "job_dir": str(job_dir), "started_at": 0},
    )
    status = await backend(make_ctx).status(ref)
    assert status.state is JobState.FAILED


async def test_validate_reports_missing_training_modules(make_ctx: Callable[..., TargetContext]) -> None:
    runner = FakeRunner(lambda call: result("3.12.9\njax,brax\n"))
    report = await LocalCpuBackend(make_ctx({"python": sys.executable}), runner=runner).validate()
    assert report.status is HealthStatus.DEGRADED
    assert "jax,brax" in report.message
    assert runner.calls[0].argv[0] == sys.executable

    runner = FakeRunner(lambda call: result("3.12.9\n\n"))
    report = await LocalCpuBackend(make_ctx({"python": sys.executable}), runner=runner).validate()
    assert report.status is HealthStatus.OK


async def test_validate_without_interpreter_is_down(make_ctx: Callable[..., TargetContext]) -> None:
    report = await LocalCpuBackend(make_ctx({"python": "/nonexistent/python"})).validate()
    assert report.status is HealthStatus.DOWN


def test_python_falls_back_to_pipeline_python_then_own_interpreter(
    make_ctx: Callable[..., TargetContext],
) -> None:
    configured = make_ctx({"python": "/opt/custom/bin/python"})
    assert LocalCpuBackend(replace(configured, pipeline_python="/opt/pipeline/bin/python")).python == (
        "/opt/custom/bin/python"
    )
    assert LocalCpuBackend(replace(make_ctx({}), pipeline_python="/opt/pipeline/bin/python")).python == (
        "/opt/pipeline/bin/python"
    )
    assert LocalCpuBackend(make_ctx({})).python == sys.executable
