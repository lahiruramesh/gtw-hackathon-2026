"""local_cpu: runs a stage as a detached process on the worker host (smoke runs, CPU evaluation).

Layout per stage: <target work dir>/<work_subdir>/<run_id>/<stage_key>-<stage8>/
    in/ out/ job.log exit_code job.json reporter.err [cancelled]

The job is started in its own session, wrapped in apps/reporter/skf_reporter.py, so it keeps running
(and keeps phoning home) if the worker restarts; status() re-attaches through the pid and exit file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from skf_api.backends._job import (
    JOB_LOG,
    LOG_CHUNK_BYTES,
    REPORTER_RELPATH,
    copy_tree,
    ingest_env,
    input_relpath,
    render_argv,
    reporter_args,
    split_lines,
)
from skf_api.backends._proc import ProcessRunner, SubprocessRunner
from skf_api.backends.base import (
    BackendKind,
    ExternalRef,
    HealthReport,
    HealthStatus,
    JobSpec,
    JobState,
    JobStatus,
    LogBatch,
    TargetContext,
)

PIPELINE_MODULES = ("jax", "brax", "mujoco", "mujoco_playground")
_PROBE = (
    "import importlib.util, sys; "
    "print(sys.version.split()[0]); "
    "print(','.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))"
)
# Only these worker variables reach the job: the worker's own environment holds database, storage
# and signing secrets that pipeline code has no business seeing.
_PASSTHROUGH_ENV = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ")
_PASSTHROUGH_PREFIXES = ("XLA_", "JAX_", "MUJOCO_", "OMP_")

_HAS_PROC = Path("/proc/self").exists()
# Jobs started by this worker process, so their exit is reaped; after a restart the pid is all we have.
_children: dict[int, subprocess.Popen[bytes]] = {}


class LocalCpuConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Local CPU")

    max_concurrent_hint: int = Field(2, ge=1, description="How many stages this host can run at once")
    python: str | None = Field(
        None,
        description="Interpreter with the pipeline's training deps (default: the worker's PIPELINE_PYTHON, "
        "then the worker's own interpreter)",
    )
    work_subdir: str = Field(
        "local",
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Sub-directory of the target work dir for job files",
    )


class LocalCpuSecret(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Local CPU (no credentials needed)")


class LocalCpuBackend:
    kind = BackendKind.LOCAL_CPU

    def __init__(
        self, ctx: TargetContext, *, runner: ProcessRunner | None = None, kill_grace_seconds: float = 10.0
    ) -> None:
        self.ctx = ctx
        self.config = LocalCpuConfig.model_validate(ctx.config)
        self.runner = runner or SubprocessRunner()
        self.kill_grace_seconds = kill_grace_seconds

    @property
    def python(self) -> str:
        return self.config.python or self.ctx.pipeline_python or sys.executable

    async def validate(self) -> HealthReport:
        python = shutil.which(self.python)
        reporter = self.ctx.pipeline_repo_dir / REPORTER_RELPATH
        if python is None:
            return HealthReport(HealthStatus.DOWN, f"pipeline interpreter not found: {self.python}")
        if not reporter.is_file() or not (self.ctx.pipeline_repo_dir / "g1pipe").is_dir():
            message = f"{self.ctx.pipeline_repo_dir} does not hold the pipeline repo (g1pipe/, reporter)"
            return HealthReport(HealthStatus.DOWN, message)
        probe = await self.runner([python, "-c", _PROBE, *PIPELINE_MODULES], timeout_s=60)
        if not probe.ok:
            return HealthReport(HealthStatus.DOWN, f"{python} failed to start: {probe.error_text}")
        version, _, missing = probe.text.strip().partition("\n")
        details = {
            "python": python,
            "python_version": version,
            "cpu_count": os.cpu_count(),
            "max_concurrent_hint": self.config.max_concurrent_hint,
        }
        if missing.strip():
            return HealthReport(
                HealthStatus.DEGRADED,
                f"{python} lacks training modules: {missing.strip()} (evaluation may still work)",
                details,
            )
        return HealthReport(HealthStatus.OK, f"Python {version}, pipeline modules importable", details)

    async def submit(self, spec: JobSpec) -> ExternalRef:
        job_dir = self._job_dir(spec)
        existing = _read_json(job_dir / "job.json")
        if existing:  # submitted before (worker restarted mid-submit)
            return self._ref(job_dir, existing)
        in_dir, out_dir = job_dir / "in", job_dir / "out"
        await asyncio.to_thread(_stage_inputs, spec, in_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        argv = render_argv(spec, [self.python], str(out_dir), str(in_dir))
        progress = str(out_dir / spec.progress_csv) if spec.progress_csv else None
        repo = self.ctx.pipeline_repo_dir
        cmd = [
            sys.executable,
            str(repo / REPORTER_RELPATH),
            *reporter_args(
                log=str(job_dir / JOB_LOG),
                exit_file=str(job_dir / "exit_code"),
                progress=progress,
                timeout_minutes=spec.timeout_minutes,
            ),
            "--",
            *argv,
        ]
        proc = await asyncio.to_thread(_spawn, cmd, repo, self._env(spec), job_dir / "reporter.err")
        _children[proc.pid] = proc
        record = {"pid": proc.pid, "started_at": time.time()}
        _write_json(job_dir / "job.json", record)
        return self._ref(job_dir, record)

    async def status(self, ref: ExternalRef) -> JobStatus:
        job_dir, pid = _job_dir_of(ref), int(ref.data["pid"])
        if (job_dir / "cancelled").exists():
            return JobStatus(JobState.CANCELLED, "cancelled")
        code = _exit_code(job_dir)
        if code is None and await self._alive(pid, job_dir):
            return JobStatus(JobState.RUNNING)
        code = code if code is not None else _exit_code(job_dir)  # it may have ended in between
        if code is None:
            return JobStatus(JobState.FAILED, "job ended without an exit code (killed or host restarted)")
        if code == 0:
            return JobStatus(JobState.SUCCEEDED)
        return JobStatus(JobState.FAILED, "timed out" if code == 124 else f"exit code {code}")

    async def fetch_logs(self, ref: ExternalRef, cursor: str | None) -> LogBatch:
        job_dir = _job_dir_of(ref)
        offset = int(cursor or 0)
        chunk, size = await asyncio.to_thread(_read_chunk, job_dir / JOB_LOG, offset)
        finished = _exit_code(job_dir) is not None or (job_dir / "cancelled").exists()
        lines, used = split_lines(chunk, final=finished and offset + len(chunk) >= size)
        return LogBatch(lines, str(offset + used))

    async def collect(self, ref: ExternalRef, dest: Path) -> None:
        job_dir = _job_dir_of(ref)
        await asyncio.to_thread(copy_tree, job_dir / "out", dest)
        if (job_dir / JOB_LOG).is_file():
            await asyncio.to_thread(shutil.copyfile, job_dir / JOB_LOG, dest / JOB_LOG)

    async def cancel(self, ref: ExternalRef) -> None:
        job_dir, pid = _job_dir_of(ref), int(ref.data["pid"])
        if _exit_code(job_dir) is None:
            (job_dir / "cancelled").touch()
        if not await self._alive(pid, job_dir):
            return
        _signal_group(pid, signal.SIGTERM)  # the reporter forwards it to the job
        deadline = time.monotonic() + self.kill_grace_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            if not await self._alive(pid, job_dir):
                return
        _signal_group(pid, signal.SIGKILL)

    async def release(self) -> None:
        """Removes the dirs of finished jobs. It is called once no stage uses the target, so their outputs
        are collected; a job that is still running (or still being submitted) keeps its dir."""
        root = self.ctx.work_dir / self.config.work_subdir
        job_dirs = await asyncio.to_thread(lambda: [d for d in root.glob("*/*") if d.is_dir()])
        for job_dir in job_dirs:
            if await self._finished(job_dir):
                await asyncio.to_thread(shutil.rmtree, job_dir, ignore_errors=True)
        await asyncio.to_thread(_remove_empty_dirs, root)

    async def _finished(self, job_dir: Path) -> bool:
        if _exit_code(job_dir) is not None:
            return True
        record = _read_json(job_dir / "job.json")
        return (
            record is not None
            and (job_dir / "cancelled").exists()
            and not await self._alive(int(record["pid"]), job_dir)
        )

    def _job_dir(self, spec: JobSpec) -> Path:
        return (
            self.ctx.work_dir
            / self.config.work_subdir
            / spec.run_id
            / f"{spec.stage_key}-{spec.stage_id[:8]}"
        )

    def _ref(self, job_dir: Path, record: dict) -> ExternalRef:
        data = {"pid": record["pid"], "job_dir": str(job_dir), "started_at": record["started_at"]}
        return ExternalRef(BackendKind.LOCAL_CPU, str(record["pid"]), data)

    def _env(self, spec: JobSpec) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if k in _PASSTHROUGH_ENV or k.startswith(_PASSTHROUGH_PREFIXES)
        }
        env.update(PYTHONPATH=str(self.ctx.pipeline_repo_dir), PYTHONUNBUFFERED="1")
        if sys.platform.startswith("linux"):
            env["MUJOCO_GL"] = "egl"  # headless rendering for evaluation videos
        env.update(spec.env)
        env.update(ingest_env(spec.ingest, spec.stage_id))
        return env

    async def _alive(self, pid: int, job_dir: Path) -> bool:
        """True if `pid` is still our job. For jobs another worker process started, the command line is
        checked too, because after a host restart the pid may belong to an unrelated process."""
        child = _children.get(pid)
        if child is not None:
            if child.poll() is None:
                return True
            del _children[pid]
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False  # someone else's process now
        cmdline = await self._cmdline(pid)
        return cmdline is not None and str(job_dir) in cmdline

    async def _cmdline(self, pid: int) -> str | None:
        if _HAS_PROC:
            return await asyncio.to_thread(_proc_cmdline, pid)
        result = await self.runner(["ps", "-p", str(pid), "-o", "command="], timeout_s=10)
        return result.text if result.ok else None


def _spawn(cmd: list[str], cwd: Path, env: dict[str, str], stderr_path: Path) -> subprocess.Popen[bytes]:
    with open(stderr_path, "ab") as err:
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=err,
            start_new_session=True,
        )


def _proc_cmdline(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return None


def _stage_inputs(spec: JobSpec, in_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    for item in spec.inputs:
        target = in_dir / input_relpath(item.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.local_path, target)


def _job_dir_of(ref: ExternalRef) -> Path:
    return Path(ref.data["job_dir"])


def _exit_code(job_dir: Path) -> int | None:
    try:
        return int((job_dir / "exit_code").read_text().strip())
    except (OSError, ValueError):
        return None


def _remove_empty_dirs(root: Path) -> None:
    for run_dir in root.glob("*"):
        with contextlib.suppress(OSError):  # not empty, or not a dir
            run_dir.rmdir()


def _read_chunk(path: Path, offset: int) -> tuple[bytes, int]:
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            f.seek(offset)
            return f.read(LOG_CHUNK_BYTES), size
    except FileNotFoundError:
        return b"", 0


def _signal_group(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)  # start_new_session made the pid the group id


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
