"""Compute backend contract.

A backend runs one pipeline stage somewhere (this machine, a Kaggle GPU kernel, an AWS GPU box)
and reports back. The orchestrator (skf_api.orchestrator) only talks to this interface, so adding
a location (GCP, Slurm, Kubernetes) means adding one module and one registry entry.

Lifecycle, driven by the orchestrator:

    validate()  -> health for the compute page (read-only, cheap, never starts paid resources)
    submit()    -> start the job, return an ExternalRef that is persisted on the stage
    status()    -> polled by the reconciler until a terminal state
    fetch_logs()-> only used when the job can't phone home (no public ingest URL)
    collect()   -> copy outputs into the local staging dir; the orchestrator uploads them
    cancel()    -> stop the job and release remote resources; must be idempotent
    release()   -> called after collect/cancel when no other stage uses the target (e.g. stop the box)

Implementations must be safe to call again after a worker restart with the same ExternalRef.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class BackendKind(enum.StrEnum):
    LOCAL_CPU = "local_cpu"
    KAGGLE = "kaggle"
    AWS_EC2 = "aws_ec2"


class JobState(enum.StrEnum):
    PROVISIONING = "provisioning"  # waiting for capacity / instance start / kernel queue
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


class HealthStatus(enum.StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InputFile:
    """A file the job needs, already downloaded by the orchestrator to `local_path`.
    `name` is the path relative to the job's input dir, e.g. "train/params.pkl"."""

    name: str
    local_path: Path
    # For backends that can mount outputs of an earlier remote job instead of uploading
    # (Kaggle kernel_sources): the ExternalRef of the stage that produced it, if any.
    source_ref: ExternalRef | None = None


@dataclass(frozen=True)
class IngestConfig:
    """Where the in-job reporter (apps/reporter/skf_reporter.py) should phone home.
    None on JobSpec means no public ingest URL: the orchestrator polls fetch_logs() instead."""

    url: str  # e.g. https://studio.example.com/ingest/v1
    token: str  # per-stage scoped token, sent as `Authorization: Bearer <token>`


@dataclass(frozen=True)
class JobSpec:
    run_id: str
    run_name: str  # human slug, e.g. "g1-stairs-v12"; unique
    stage_id: str
    stage_key: str  # manifest stage id, e.g. "train"
    kind: str  # "train" | "evaluate"
    # Rendered command, run from the pipeline repo root with PYTHONPATH=<repo root>.
    # argv[0] is always the literal "python"; each backend substitutes its interpreter
    # (local: settings.pipeline_python, AWS: "uv run python", Kaggle: sys.executable).
    # Placeholders are already resolved EXCEPT these two, which backends replace:
    #   {out_dir}   the directory the job must write its outputs to
    #   {input_dir} the directory holding `inputs` (layout: <input_dir>/<InputFile.name>)
    argv: list[str]
    inputs: list[InputFile] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    ingest: IngestConfig | None = None
    progress_csv: str | None = "progress.csv"  # relative to out_dir; tailed into metrics
    git_sha: str | None = None
    timeout_minutes: int = 24 * 60


@dataclass(frozen=True)
class ExternalRef:
    """Persisted as JSON on the stage (stage.external_ref). Must be enough to resume after restart."""

    backend: BackendKind
    job_id: str  # pid / kernel slug / tmux session name
    data: dict[str, Any] = field(default_factory=dict)  # e.g. instance_id, remote paths, log offsets
    url: str | None = None  # link shown in the UI (Kaggle kernel page, EC2 console)

    def to_json(self) -> dict[str, Any]:
        return {"backend": self.backend.value, "job_id": self.job_id, "data": self.data, "url": self.url}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ExternalRef:
        return cls(BackendKind(d["backend"]), d["job_id"], dict(d.get("data") or {}), d.get("url"))


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    message: str | None = None
    progress: float | None = None  # 0..1 if known
    gpu_seconds: float | None = None  # billable accelerator time so far
    ref: ExternalRef | None = None  # updated ref to persist (e.g. new IP, log offset)


@dataclass(frozen=True)
class LogBatch:
    lines: list[str]
    cursor: str | None  # opaque; pass back to fetch_logs next time


@dataclass(frozen=True)
class HealthReport:
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class BackendError(Exception):
    """Expected, user-visible failure (bad credentials, no capacity, quota). Message is shown in the UI."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class ComputeBackend(Protocol):
    kind: BackendKind

    async def validate(self) -> HealthReport: ...
    async def submit(self, spec: JobSpec) -> ExternalRef: ...
    async def status(self, ref: ExternalRef) -> JobStatus: ...
    async def fetch_logs(self, ref: ExternalRef, cursor: str | None) -> LogBatch: ...
    async def collect(self, ref: ExternalRef, dest: Path) -> None: ...
    async def cancel(self, ref: ExternalRef) -> None: ...
    async def release(self) -> None: ...


@dataclass(frozen=True)
class TargetContext:
    """Everything a backend gets at construction. Built by the orchestrator from the compute_targets row."""

    target_id: str
    name: str
    config: dict[str, Any]  # non-secret, validated by the backend's Config model
    secret: dict[str, Any]  # decrypted secret JSON ({} if none); never log it
    work_dir: Path  # scratch dir for this target on the worker host
    pipeline_repo_dir: Path  # repo root holding g1pipe/, scripts/, skills/, apps/reporter/
    # Interpreter with the pipeline's training deps (settings.pipeline_python); local jobs run on it.
    pipeline_python: str | None = None
