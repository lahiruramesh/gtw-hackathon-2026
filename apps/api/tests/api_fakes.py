"""Test doubles for everything outside the API process: the identity provider (Better Auth's JWKS),
compute backends and the job queue."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jwt.algorithms import OKPAlgorithm

from skf_api.backends import config_schema
from skf_api.backends.base import (
    BackendError,
    BackendKind,
    ExternalRef,
    HealthReport,
    HealthStatus,
    JobSpec,
    JobState,
    JobStatus,
    LogBatch,
)
from skf_api.context import AppContext
from skf_api.modules.compute.models import ComputeTarget
from skf_api.orchestrator import executor, state_machine

ISSUER, AUDIENCE, KID = "skf-skill-studio", "skf-api", "test-key-1"
USERS = {role: f"user-{role}" for role in ("admin", "ml_engineer", "operator", "safety_reviewer", "viewer")}


class Keys:
    def __init__(self) -> None:
        self.private = Ed25519PrivateKey.generate()
        self.stranger = Ed25519PrivateKey.generate()  # a key the JWKS does not publish
        jwk = OKPAlgorithm.to_jwk(self.private.public_key(), as_dict=True)
        self.jwks = {"keys": [{**jwk, "kid": KID, "alg": "EdDSA", "use": "sig"}]}

    def token(
        self, role: str | None = "viewer", *, sub: str | None = None, key: Any = None, **claims: Any
    ) -> str:
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": sub or USERS.get(role or "", "user-x"),
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + 900,
            "email": f"{role}@skf.test",
            "name": f"{role} user",
        }
        if role is not None:
            payload["role"] = role
        payload.update(claims)
        return jwt.encode(payload, key or self.private, algorithm="EdDSA", headers={"kid": KID})

    def headers(self, role: str, *, sub: str | None = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(role, sub=sub)}"}


def strict_results(*, crossed: bool = True, fell: bool = False) -> list[dict[str, Any]]:
    """96 strict-test crossings (5 step heights) as g1pipe.stairs_eval writes them."""
    rises = [0.03, 0.05, 0.08, 0.10, 0.12]
    return [
        {
            "rise_m": rises[i % len(rises)],
            "kind": "pyramid",
            "crossed": crossed,
            "fell": fell,
            "reached_centre": crossed,
            "tilt_deg": [4.0, 9.0],
            "pelvis_rel_m": [0.7, 0.74],
            "start": [0, 0],
        }
        for i in range(96)
    ]


def train_outputs(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "params.pkl").write_bytes(b"final-params")
    (dest / "config.json").write_text(json.dumps({"env": {"leg_action_scale": 1.0}}))
    for step in (1000, 2000):
        (dest / f"ckpt_{step:011d}.pkl").write_bytes(f"ckpt-{step}".encode())
    (dest / "progress.csv").write_text(
        "step,wall_s,eval/episode_reward,eval/episode_crossed,eval/episode_reward/alive,eval/episode_reward_std\n"
        "0,10.0,1.0,0.1,5.0,0.5\n1000,20.0,2.0,0.4,6.0,0.4\n2000,30.0,3.5,0.8,7.0,0.3\n"
    )


@dataclass
class FakeJob:
    spec: JobSpec
    target_id: str
    state: JobState = JobState.RUNNING
    logs: list[str] = field(default_factory=list)
    gpu_seconds: float = 0.0
    message: str | None = None


@dataclass
class FakeWorld:
    """Everything the fake backends know, shared across instances like a real remote system."""

    jobs: dict[str, FakeJob] = field(default_factory=dict)
    cancelled: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    submitted: list[JobSpec] = field(default_factory=list)
    strict_pass: bool = True
    health: HealthReport = field(default_factory=lambda: HealthReport(HealthStatus.OK, "all good"))
    # Failure injection and hooks that run inside a backend call (while the orchestrator awaits it).
    submit_error: BackendError | None = None
    cancel_errors: list[BackendError] = field(default_factory=list)  # raised by the next cancel() calls
    during_submit: Callable[[JobSpec], Awaitable[None]] | None = None
    during_status: Callable[[], Awaitable[None]] | None = None
    extra_outputs: Callable[[Path], None] | None = None  # adds files to a collected train stage

    def job(self, stage_key: str) -> FakeJob:
        """The latest job submitted for a stage key (e.g. "train")."""
        return next(j for j in reversed(self.jobs.values()) if j.spec.stage_key == stage_key)

    def finish(self, stage_key: str, state: JobState = JobState.SUCCEEDED, **kwargs: Any) -> FakeJob:
        job = self.job(stage_key)
        job.state = state
        for name, value in kwargs.items():
            setattr(job, name, value)
        return job


class FakeBackend:
    def __init__(self, world: FakeWorld, target: ComputeTarget):
        self.world = world
        self.kind = target.kind
        self.target_id = str(target.id)

    async def validate(self) -> HealthReport:
        return self.world.health

    async def submit(self, spec: JobSpec) -> ExternalRef:
        for item in spec.inputs:
            assert item.local_path.is_file(), f"input {item.name} was not downloaded"
        if self.world.submit_error is not None:
            raise self.world.submit_error
        job_id = f"job-{len(self.world.jobs) + 1}"
        self.world.jobs[job_id] = FakeJob(spec, self.target_id)
        self.world.submitted.append(spec)
        if self.world.during_submit is not None:
            await self.world.during_submit(spec)
        return ExternalRef(
            BackendKind(self.kind), job_id, {"stage": spec.stage_id}, url=f"https://jobs.test/{job_id}"
        )

    async def status(self, ref: ExternalRef) -> JobStatus:
        job = self.world.jobs[ref.job_id]
        status = JobStatus(job.state, message=job.message, gpu_seconds=job.gpu_seconds)
        if self.world.during_status is not None:
            await self.world.during_status()
        return status

    async def fetch_logs(self, ref: ExternalRef, cursor: str | None) -> LogBatch:
        logs = self.world.jobs[ref.job_id].logs
        start = int(cursor or 0)
        return LogBatch(logs[start:], str(len(logs)))

    async def collect(self, ref: ExternalRef, dest: Path) -> None:
        job = self.world.jobs[ref.job_id]
        if job.spec.kind == "train":
            train_outputs(dest)
            if self.world.extra_outputs is not None:
                self.world.extra_outputs(dest)
        else:
            results = strict_results(crossed=self.world.strict_pass, fell=not self.world.strict_pass)
            (dest / "strict.json").write_text(json.dumps(results))

    async def cancel(self, ref: ExternalRef) -> None:
        if self.world.cancel_errors:
            raise self.world.cancel_errors.pop(0)
        self.world.jobs[ref.job_id].state = JobState.CANCELLED
        self.world.cancelled.append(ref.job_id)

    async def release(self) -> None:
        self.world.released.append(self.target_id)


class FakeBackendFactory:
    def __init__(self, world: FakeWorld):
        self.world = world

    def for_target(self, target: ComputeTarget) -> FakeBackend:
        return FakeBackend(self.world, target)

    def schema(self, kind: BackendKind) -> dict[str, Any]:
        return config_schema(kind)


class RecordingJobQueue:
    """Records enqueued jobs; `drain` runs them in order, like a worker with one slot."""

    def __init__(self) -> None:
        self.pending: list[tuple[str, uuid.UUID]] = []

    async def advance_run(self, run_id: uuid.UUID) -> None:
        self.pending.append(("advance_run", run_id))

    async def execute_stage(self, stage_id: uuid.UUID, *, defer_seconds: float = 0) -> None:
        self.pending.append(("execute_stage", stage_id))

    async def cancel_run(self, run_id: uuid.UUID) -> None:
        self.pending.append(("cancel_run", run_id))

    async def drain(self, ctx: AppContext, *, limit: int = 50) -> list[str]:
        ran: list[str] = []
        while self.pending:
            assert len(ran) < limit, f"job loop: {ran}"
            name, entity_id = self.pending.pop(0)
            ran.append(name)
            if name == "advance_run":
                await state_machine.advance_run(ctx, entity_id)
            elif name == "execute_stage":
                await executor.execute_stage(ctx, entity_id)
            else:
                await state_machine.cancel_run(ctx, entity_id)
        return ran
