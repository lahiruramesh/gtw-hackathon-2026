"""Run duration, GPU hours and cost from a target's measured throughput (SPEC §4, §9.1 Estimate).

Train time = timesteps / steps_per_second + the target's fixed overhead (install, JIT, evals, provisioning).
Evaluate stages use their manifest `estimate_minutes`. Only stages on a GPU target count as GPU hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skf_api.backends.base import BackendKind
from skf_api.modules.compute.backends import KINDS_NEEDING_SECRET
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.compute.service import Usage
from skf_api.modules.runs.models import StageKind
from skf_api.modules.runs.schemas import Estimate
from skf_api.skills_registry.manifest import Manifest, StageDef

DEFAULT_EVALUATE_MINUTES = 10.0
OPERATOR_ROLE = "operator"


@dataclass(frozen=True)
class EstimateInput:
    manifest: Manifest
    params: dict[str, Any]  # validated, defaults applied
    target: ComputeTarget
    usage: Usage
    role: str
    allow_local_training: bool  # ENVIRONMENT=development


def is_smoke(params: dict[str, Any]) -> bool:
    return bool(params.get("smoke"))


def _stage_minutes(stage: StageDef, params: dict[str, Any], target: ComputeTarget) -> float:
    if stage.estimate_minutes is not None:
        return stage.estimate_minutes
    timesteps = params.get("timesteps")
    if stage.kind is StageKind.TRAIN and isinstance(timesteps, int):
        return timesteps / target.steps_per_second / 60.0 + target.overhead_minutes
    return DEFAULT_EVALUATE_MINUTES


def estimate_run(inp: EstimateInput) -> Estimate:
    target = inp.target
    train_minutes = sum(_stage_minutes(s, inp.params, target) for s in inp.manifest.train_stages)
    other_minutes = sum(_stage_minutes(s, inp.params, target) for s in inp.manifest.evaluate_stages)
    gpu_minutes = sum(
        _stage_minutes(s, inp.params, target) for s in inp.manifest.pipeline if s.runs_gpu_target
    )
    gpu_hours = 0.0 if target.kind is BackendKind.LOCAL_CPU else gpu_minutes / 60.0
    cost = gpu_hours * target.cost_per_gpu_hour

    reasons: list[str] = []
    if inp.role == OPERATOR_ROLE and gpu_hours > target.max_unapproved_gpu_hours:
        reasons.append(
            f"{gpu_hours:.1f} GPU-h exceeds the {target.max_unapproved_gpu_hours:g} GPU-h an operator "
            f"may launch on {target.name} without approval"
        )

    warnings: list[str] = []
    if not target.enabled:
        warnings.append(f"{target.name} is disabled")
    if target.kind in KINDS_NEEDING_SECRET and not target.has_secret:
        warnings.append(f"{target.name} has no credentials set")
    quota_left = inp.usage.quota_left(target)
    if quota_left is not None and gpu_hours > quota_left:
        warnings.append(
            f"Needs {gpu_hours:.1f} GPU-h but only {quota_left:.1f} GPU-h of the weekly quota is left"
        )
    if target.kind is BackendKind.LOCAL_CPU and not is_smoke(inp.params) and inp.manifest.train_stages:
        warnings.append(
            "Full training on the local CPU target takes days; use a GPU target or the smoke preset"
            if inp.allow_local_training
            else "Only smoke runs may train on the local CPU target"
        )
    if inp.usage.active_stages >= target.max_concurrent:
        warnings.append(f"{target.name} is busy; the run will wait for a free slot")

    return Estimate(
        train_minutes=round(train_minutes, 1),
        total_minutes=round(train_minutes + other_minutes, 1),
        gpu_hours=round(gpu_hours, 3),
        cost=round(cost, 2),
        needs_approval=bool(reasons),
        reasons=reasons,
        warnings=warnings,
    )
