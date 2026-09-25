"""The three targets this repo's experiments ran on, created once by `skf-api seed-targets`.

Remote targets start disabled: an admin sets their credentials (and the Kaggle username) first.
Throughput and overhead are the measured numbers from docs/effort_log.csv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.backends.base import BackendKind
from skf_api.modules.audit import service as audit
from skf_api.modules.compute.models import ComputeTarget

LOCAL_CPU = "Local CPU"
KAGGLE_T4 = "Kaggle T4"
AWS_L40S = "AWS L40S"
AWS_L40S_COST_PER_GPU_HOUR = 1.86  # g6e.xlarge on demand, us-east-1


@dataclass(frozen=True)
class TargetSeed:
    name: str
    kind: BackendKind
    description: str
    enabled: bool
    gpu_label: str | None
    steps_per_second: float
    overhead_minutes: float
    cost_per_gpu_hour: float
    max_concurrent: int
    max_unapproved_gpu_hours: float
    weekly_quota_gpu_hours: float | None = None
    config: dict[str, Any] = field(default_factory=dict)


def default_seeds(kaggle_username: str | None = None) -> list[TargetSeed]:
    kaggle_config: dict[str, Any] = {
        "accelerator": "NvidiaTeslaT4",
        "kernel_prefix": "skf",
        "enable_internet": True,
    }
    if kaggle_username:
        kaggle_config["username"] = kaggle_username
    return [
        TargetSeed(
            name=LOCAL_CPU,
            kind=BackendKind.LOCAL_CPU,
            enabled=True,
            gpu_label=None,
            description="The worker host's CPU: smoke runs and cross-engine evaluation.",
            steps_per_second=250,
            overhead_minutes=1,
            cost_per_gpu_hour=0,
            max_concurrent=2,
            max_unapproved_gpu_hours=0,
            config={"max_concurrent_hint": 2},
        ),
        TargetSeed(
            name=KAGGLE_T4,
            kind=BackendKind.KAGGLE,
            enabled=False,
            gpu_label="1x T4",
            description="Free Kaggle T4 kernels (30 GPU-hours per week, phone-verified account).",
            steps_per_second=36_000,
            overhead_minutes=8,
            cost_per_gpu_hour=0,
            max_concurrent=2,
            max_unapproved_gpu_hours=4,
            weekly_quota_gpu_hours=30,
            config=kaggle_config,
        ),
        TargetSeed(
            name=AWS_L40S,
            kind=BackendKind.AWS_EC2,
            enabled=False,
            gpu_label="1x L40S 48 GB",
            description="The g1-train box from scripts/aws_box.sh (g6e.xlarge on demand, idle auto-stop).",
            steps_per_second=100_000,
            overhead_minutes=12,
            cost_per_gpu_hour=AWS_L40S_COST_PER_GPU_HOUR,
            max_concurrent=1,
            max_unapproved_gpu_hours=2,
            config={
                "region": "us-east-1",
                "instance_name": "g1-train",
                # the same box in other zones: g6e capacity comes and goes per availability zone
                "fallback_instance_names": ["g1-train-2", "g1-train-3"],
                "instance_type": "g6e.xlarge",
                "ami_id": "ami-028e28e7fc9d87d00",
                "subnet_id": "subnet-0dc6bf369fb078ba3",
                "security_group_id": "sg-01c410eb3570de544",
                "key_name": "g1-train",
                # No named profile: the worker always runs in the cloud and uses the host's IAM role.
            },
        ),
    ]


async def seed_targets(session: AsyncSession, seeds: list[TargetSeed]) -> list[str]:
    """Creates missing targets by name; existing ones (possibly edited by an admin) are left alone."""
    existing = set(await session.scalars(sa.select(ComputeTarget.name)))
    created = []
    for seed in seeds:
        if seed.name in existing:
            continue
        target = ComputeTarget(
            name=seed.name,
            kind=seed.kind,
            description=seed.description,
            enabled=seed.enabled,
            config=seed.config,
            gpu_label=seed.gpu_label,
            steps_per_second=seed.steps_per_second,
            overhead_minutes=seed.overhead_minutes,
            cost_per_gpu_hour=seed.cost_per_gpu_hour,
            weekly_quota_gpu_hours=seed.weekly_quota_gpu_hours,
            max_unapproved_gpu_hours=seed.max_unapproved_gpu_hours,
            max_concurrent=seed.max_concurrent,
        )
        session.add(target)
        await session.flush()
        audit.record(
            session,
            audit.SYSTEM_ACTOR,
            "target.create",
            "compute_target",
            target.id,
            {"name": seed.name, "kind": seed.kind.value, "seeded": True},
        )
        created.append(seed.name)
    await session.commit()
    return created
