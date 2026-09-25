"""Render a manifest stage into an argv list (never a shell string) plus the input files it needs.

`{param}` placeholders are replaced with the validated param value; `{out_dir}` and `{input_dir}` are left
for the backend. `flags` map a param to a CLI flag: true -> flag, other non-null values -> flag + value,
checkpoint params -> flag + `{input_dir}/parent/<file>` with the checkpoint staged as an input.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from skf_api.skills_registry.manifest import PLACEHOLDER, Manifest, ParamType, StageDef


class RenderError(Exception):
    pass


@dataclass(frozen=True)
class CheckpointFile:
    artifact_id: uuid.UUID
    filename: str  # e.g. "ckpt_00253624320.pkl" or "params.pkl"


@dataclass(frozen=True)
class StageOutputInput:
    """An output of an earlier stage of the same run, e.g. "train/params.pkl"."""

    name: str

    @property
    def stage_key(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def filename(self) -> str:
        return self.name.split("/", 1)[1]


@dataclass(frozen=True)
class ArtifactInput:
    """A specific stored artifact (warm-start checkpoint, or the checkpoint under evaluation)."""

    name: str
    artifact_id: uuid.UUID


type StageInputRef = StageOutputInput | ArtifactInput


@dataclass(frozen=True)
class RenderedStage:
    argv: list[str]
    inputs: list[StageInputRef]


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_stage(
    manifest: Manifest,
    stage: StageDef,
    params: dict[str, Any],
    checkpoints: dict[str, CheckpointFile] | None = None,
    evaluate_checkpoint: CheckpointFile | None = None,
) -> RenderedStage:
    """`params` must already be validated. `checkpoints` resolves checkpoint params to files;
    `evaluate_checkpoint` points an evaluate stage at a periodic checkpoint instead of the final params."""
    if stage.argv is None:
        raise RenderError(f"stage '{stage.id}' has no command")
    checkpoints = checkpoints or {}

    def substitute(token: str) -> str:
        def repl(match: Any) -> str:
            name = match.group(1)
            if name in ("out_dir", "input_dir"):
                return match.group(0)
            value = params.get(name)
            if value is None:
                raise RenderError(f"param '{name}' is required by stage '{stage.id}'")
            return _text(value)

        return PLACEHOLDER.sub(repl, token)

    argv = [substitute(token) for token in stage.argv]
    inputs: list[StageInputRef] = [StageOutputInput(name) for name in stage.inputs]

    for param, flag in stage.flags.items():
        value = params.get(param)
        if value is None or value is False:
            continue
        if manifest.params[param].type is ParamType.CHECKPOINT:
            ckpt = checkpoints.get(param)
            if ckpt is None:
                raise RenderError(f"checkpoint for '{param}' was not resolved")
            name = f"parent/{ckpt.filename}"
            argv += [flag, f"{{input_dir}}/{name}"]
            inputs.append(ArtifactInput(name, ckpt.artifact_id))
        elif value is True:
            argv.append(flag)
        else:
            argv += [flag, _text(value)]

    if evaluate_checkpoint is not None and evaluate_checkpoint.filename != "params.pkl":
        argv, inputs = _point_at_checkpoint(stage, argv, inputs, evaluate_checkpoint)
    return RenderedStage(argv=argv, inputs=inputs)


def _point_at_checkpoint(
    stage: StageDef, argv: list[str], inputs: list[StageInputRef], ckpt: CheckpointFile
) -> tuple[list[str], list[StageInputRef]]:
    # The checkpoint is staged next to the final params.pkl/config.json of the same stage: the pipeline's
    # loaders read obs sizes and env config from those siblings (g1pipe.train.load_policy, stairs_eval).
    params_input = next((i for i in stage.inputs if i.endswith("/params.pkl")), None)
    if params_input is None:
        raise RenderError(f"stage '{stage.id}' has no params.pkl input to evaluate a checkpoint with")
    ckpt_name = f"{params_input.rsplit('/', 1)[0]}/{ckpt.filename}"
    old, new = f"{{input_dir}}/{params_input}", f"{{input_dir}}/{ckpt_name}"
    if not any(old in token for token in argv):
        raise RenderError(f"stage '{stage.id}' argv does not reference {old}")
    return [token.replace(old, new) for token in argv], [*inputs, ArtifactInput(ckpt_name, ckpt.artifact_id)]
