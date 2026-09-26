"""Skill manifest (`skills/<dir>/skill.yaml`), exactly the shape in SPEC §7."""

from __future__ import annotations

import enum
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skf_api.modules.runs.models import RunsOn, StageKind
from skf_api.modules.skills.models import SkillCategory, SkillStatus

SLUG = r"^[a-z0-9][a-z0-9-]{1,62}$"
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Resolved by the backend, not from params.
BACKEND_PLACEHOLDERS = frozenset({"out_dir", "input_dir"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParamType(enum.StrEnum):
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"
    CHECKPOINT = "checkpoint"  # value is an artifact id; rendered as an input file


class ParamSpec(_Strict):
    type: ParamType = ParamType.STRING
    title: str | None = None
    description: str | None = None
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    enum: list[Any] | None = None
    nullable: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> ParamSpec:
        if self.type is ParamType.CHECKPOINT and not self.nullable:
            raise ValueError("checkpoint params must be nullable (a run may start from scratch)")
        if self.default is None and not self.nullable:
            raise ValueError("a param without a default must be nullable")
        return self


class Preset(_Strict):
    id: str = Field(pattern=SLUG)
    name: str
    description: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class StageDef(_Strict):
    id: str = Field(pattern=SLUG)
    kind: StageKind
    title: str
    runs_on: RunsOn = RunsOn.TARGET
    argv: list[str] | None = None
    flags: dict[str, str] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    progress_csv: str | None = None
    suite: str | None = None
    estimate_minutes: float | None = None
    timeout_minutes: int | None = None

    @model_validator(mode="after")
    def _by_kind(self) -> StageDef:
        if self.kind is StageKind.GATE:
            if self.argv or self.flags or self.inputs:
                raise ValueError(
                    "gate stages are evaluated by the orchestrator and take no argv/flags/inputs"
                )
            return self
        if not self.argv or self.argv[0] != "python":
            raise ValueError("argv must be a list starting with the literal 'python'")
        for name in self.inputs:
            if "/" not in name or name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"input '{name}' must be '<stage id>/<relative path>'")
        return self

    @property
    def runs_gpu_target(self) -> bool:
        return self.kind is not StageKind.GATE and self.runs_on is RunsOn.TARGET


type GateOp = Literal["<", "<=", ">", ">=", "==", "!="]
# simulation: the policy works in simulation under nominal conditions (first gate of docs/pipeline.md §6).
# release: the hardware release bar; passing it needs every criterion, the simulation ones included.
type GateLevel = Literal["simulation", "release"]


class GateCriterion(_Strict):
    metric: str
    op: GateOp
    value: float
    label: str | None = None
    level: GateLevel = "release"

    @property
    def display_label(self) -> str:
        return self.label or self.metric


class Headline(_Strict):
    label: str
    metric: str
    unit: str | None = None
    scale: float | None = None
    digits: int | None = None


class MetricsSpec(_Strict):
    keys: list[str] = Field(default_factory=lambda: ["eval/*"])
    primary: str | None = None


class Manifest(_Strict):
    id: str = Field(pattern=SLUG)
    name: str
    summary: str
    description: str = ""
    robot: str
    category: SkillCategory
    method: str
    status: SkillStatus = SkillStatus.ACTIVE
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    presets: list[Preset] = Field(default_factory=list)
    pipeline: list[StageDef]
    gate: list[GateCriterion] = Field(default_factory=list)
    headline: list[Headline] = Field(default_factory=list)
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)

    @field_validator("params")
    @classmethod
    def _param_names(cls, params: dict[str, ParamSpec]) -> dict[str, ParamSpec]:
        for name in params:
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", name) or name in BACKEND_PLACEHOLDERS:
                raise ValueError(f"invalid param name '{name}'")
        return params

    @model_validator(mode="after")
    def _pipeline(self) -> Manifest:
        ids = [s.id for s in self.pipeline]
        if len(set(ids)) != len(ids):
            raise ValueError("pipeline stage ids must be unique")
        if not self.pipeline or self.pipeline[-1].kind is not StageKind.GATE:
            raise ValueError("the pipeline must end with a gate stage")
        if sum(s.kind is StageKind.GATE for s in self.pipeline) != 1:
            raise ValueError("the pipeline must have exactly one gate stage")
        seen: set[str] = set()
        for stage in self.pipeline:
            for name in stage.inputs:
                if name.split("/", 1)[0] not in seen:
                    raise ValueError(f"stage '{stage.id}' input '{name}' must come from an earlier stage")
            for placeholder in PLACEHOLDER.findall(" ".join(stage.argv or [])):
                if placeholder not in self.params and placeholder not in BACKEND_PLACEHOLDERS:
                    raise ValueError(f"stage '{stage.id}' uses unknown placeholder '{{{placeholder}}}'")
            for param in stage.flags:
                if param not in self.params:
                    raise ValueError(f"stage '{stage.id}' flag for unknown param '{param}'")
            seen.add(stage.id)
        preset_ids = [p.id for p in self.presets]
        if len(set(preset_ids)) != len(preset_ids):
            raise ValueError("preset ids must be unique")
        return self

    def stage(self, stage_id: str) -> StageDef | None:
        return next((s for s in self.pipeline if s.id == stage_id), None)

    def preset(self, preset_id: str) -> Preset | None:
        return next((p for p in self.presets if p.id == preset_id), None)

    @property
    def train_stages(self) -> list[StageDef]:
        return [s for s in self.pipeline if s.kind is StageKind.TRAIN]

    @property
    def evaluate_stages(self) -> list[StageDef]:
        return [s for s in self.pipeline if s.kind is StageKind.EVALUATE]

    def params_schema(self) -> dict[str, Any]:
        """JSON Schema for the run form (checkpoint params are artifact ids, marked with `x-kind`)."""
        properties: dict[str, Any] = {}
        for name, spec in self.params.items():
            prop: dict[str, Any] = {"title": spec.title or name, "default": spec.default}
            if spec.description:
                prop["description"] = spec.description
            if spec.type is ParamType.CHECKPOINT:
                json_type = "string"
                prop["format"] = "uuid"
                prop["x-kind"] = "checkpoint"
            else:
                json_type = spec.type.value
            prop["type"] = [json_type, "null"] if spec.nullable else json_type
            if spec.minimum is not None:
                prop["minimum"] = spec.minimum
            if spec.maximum is not None:
                prop["maximum"] = spec.maximum
            if spec.enum is not None:
                prop["enum"] = [*spec.enum, None] if spec.nullable else list(spec.enum)
            properties[name] = prop
        return {"type": "object", "properties": properties, "additionalProperties": False}
