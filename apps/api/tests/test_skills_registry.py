"""Manifest loading, params validation and argv rendering (SPEC §7). Pure: no services needed."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from skf_api.settings import REPO_ROOT
from skf_api.skills_registry.loader import load_skills
from skf_api.skills_registry.manifest import Manifest
from skf_api.skills_registry.metrics import MetricFilter
from skf_api.skills_registry.params import ParamsError, validate_params
from skf_api.skills_registry.render import (
    ArtifactInput,
    CheckpointFile,
    RenderError,
    StageOutputInput,
    render_stage,
)

MANIFEST: dict[str, Any] = {
    "id": "toy-skill",
    "name": "Toy",
    "summary": "A skill for tests",
    "robot": "unitree-g1",
    "category": "locomotion",
    "method": "rl-ppo",
    "params": {
        "timesteps": {"type": "integer", "default": 1000, "minimum": 100, "maximum": 10**9},
        "smoke": {"type": "boolean", "default": False},
        "lr": {"type": "number", "default": None, "nullable": True, "minimum": 1e-6},
        "scan": {"enum": ["uniform", "camera"], "default": "uniform"},
        "init_from": {"type": "checkpoint", "default": None, "nullable": True},
    },
    "presets": [{"id": "smoke", "name": "Smoke", "params": {"smoke": True, "timesteps": 100}}],
    "pipeline": [
        {
            "id": "train",
            "kind": "train",
            "title": "Train",
            "runs_on": "target",
            "argv": [
                "python",
                "-m",
                "g1pipe.train",
                "--out",
                "{out_dir}",
                "--timesteps",
                "{timesteps}",
                "--tag=t{timesteps}",
            ],
            "flags": {"smoke": "--smoke", "lr": "--lr", "scan": "--scan-model", "init_from": "--init-from"},
        },
        {
            "id": "evaluate",
            "kind": "evaluate",
            "title": "Eval",
            "runs_on": "local",
            "argv": ["python", "scripts/eval_suite.py", "{input_dir}/train/params.pkl", "--out", "{out_dir}"],
            "inputs": ["train/params.pkl", "train/config.json"],
        },
        {"id": "gate", "kind": "gate", "title": "Gate"},
    ],
    "gate": [{"metric": "evaluate.x", "op": ">=", "value": 1}],
}


def manifest(**changes: Any) -> Manifest:
    return Manifest.model_validate({**MANIFEST, **changes})


def test_repo_skills_load() -> None:
    result = load_skills(REPO_ROOT / "skills")
    assert result.errors == []
    assert set(result.skills) == {"g1-step-length", "g1-stairs", "g1-stairs-bench", "ring-pick-drop"}
    stairs = result.skills["g1-stairs"].manifest
    assert [s.id for s in stairs.pipeline] == ["train", "evaluate", "gate"]
    assert stairs.params_schema()["properties"]["init_from"]["x-kind"] == "checkpoint"


def test_broken_manifests_are_reported_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "skill.yaml").write_text(yaml.safe_dump(MANIFEST))
    (tmp_path / "bad_yaml").mkdir()
    (tmp_path / "bad_yaml" / "skill.yaml").write_text("id: [unclosed")
    (tmp_path / "no_gate").mkdir()
    (tmp_path / "no_gate" / "skill.yaml").write_text(
        yaml.safe_dump({**MANIFEST, "pipeline": MANIFEST["pipeline"][:2]})
    )
    result = load_skills(tmp_path)
    assert list(result.skills) == ["toy-skill"]
    errors = {e.file.split("/")[-2]: e.message for e in result.errors}
    assert "invalid YAML" in errors["bad_yaml"]
    assert "must end with a gate stage" in errors["no_gate"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {
                "pipeline": [
                    {**MANIFEST["pipeline"][0], "argv": ["python", "{nope}"]},
                    MANIFEST["pipeline"][2],
                ]
            },
            "unknown placeholder",
        ),
        (
            {
                "pipeline": [
                    {**MANIFEST["pipeline"][1], "inputs": ["train/params.pkl"]},
                    MANIFEST["pipeline"][2],
                ]
            },
            "must come from an earlier stage",
        ),
        (
            {
                "pipeline": [
                    {**MANIFEST["pipeline"][0], "argv": ["bash", "-c", "rm -rf /"]},
                    MANIFEST["pipeline"][2],
                ]
            },
            "starting with the literal 'python'",
        ),
        ({"params": {"x": {"type": "integer"}}}, "must be nullable"),
        ({"surprise": True}, "Extra inputs"),
    ],
)
def test_manifest_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        manifest(**change)


def test_params_defaults_and_coercion() -> None:
    params = validate_params(manifest(), {"timesteps": 2e3, "lr": 1})
    assert params == {"timesteps": 2000, "smoke": False, "lr": 1.0, "scan": "uniform", "init_from": None}


@pytest.mark.parametrize(
    ("params", "field", "message"),
    [
        ({"timesteps": 10}, "timesteps", "must be >= 100"),
        ({"timesteps": 1.5}, "timesteps", "must be an integer"),
        ({"timesteps": True}, "timesteps", "must be an integer"),
        ({"smoke": "yes"}, "smoke", "must be a boolean"),
        ({"scan": "lidar"}, "scan", "must be one of uniform, camera"),
        ({"lr": 0.0}, "lr", "must be >= 1e-06"),
        ({"lr": float("nan")}, "lr", "must be a number"),
        ({"init_from": "not-a-uuid"}, "init_from", "checkpoint artifact id"),
        ({"timesteps": None}, "timesteps", "is required"),
        ({"extra": 1}, "extra", "unknown parameter"),
    ],
)
def test_params_rejected(params: dict[str, Any], field: str, message: str) -> None:
    with pytest.raises(ParamsError) as info:
        validate_params(manifest(), params)
    assert message in info.value.errors[field]


def test_render_train_with_flags_and_warm_start() -> None:
    m = manifest()
    ckpt = CheckpointFile(uuid.uuid4(), "ckpt_00000001000.pkl")
    params = validate_params(m, {"smoke": True, "lr": 0.0001, "init_from": str(ckpt.artifact_id)})
    rendered = render_stage(m, m.pipeline[0], params, {"init_from": ckpt})
    assert rendered.argv == [
        "python",
        "-m",
        "g1pipe.train",
        "--out",
        "{out_dir}",
        "--timesteps",
        "1000",
        "--tag=t1000",
        "--smoke",
        "--lr",
        "0.0001",
        "--scan-model",
        "uniform",
        "--init-from",
        "{input_dir}/parent/ckpt_00000001000.pkl",
    ]
    assert rendered.inputs == [ArtifactInput("parent/ckpt_00000001000.pkl", ckpt.artifact_id)]


def test_render_is_not_a_shell() -> None:
    m = manifest(params={**MANIFEST["params"], "scan": {"type": "string", "default": "x; rm -rf / #"}})
    rendered = render_stage(m, m.pipeline[0], validate_params(m, {}))
    assert "x; rm -rf / #" in rendered.argv  # one argv element, never interpreted


def test_render_evaluate_a_checkpoint() -> None:
    m = manifest()
    ckpt = CheckpointFile(uuid.uuid4(), "ckpt_00000002000.pkl")
    rendered = render_stage(m, m.pipeline[1], validate_params(m, {}), evaluate_checkpoint=ckpt)
    assert rendered.argv[2] == "{input_dir}/train/ckpt_00000002000.pkl"
    assert rendered.inputs == [
        StageOutputInput("train/params.pkl"),
        StageOutputInput("train/config.json"),
        ArtifactInput("train/ckpt_00000002000.pkl", ckpt.artifact_id),
    ]


def test_render_needs_resolved_checkpoints() -> None:
    m = manifest()
    params = validate_params(m, {"init_from": str(uuid.uuid4())})
    with pytest.raises(RenderError, match="not resolved"):
        render_stage(m, m.pipeline[0], params)


def test_metric_filter_defaults() -> None:
    default = MetricFilter(manifest().metrics)
    assert default.accepts("eval/episode_reward")
    assert not default.accepts("eval/episode_reward_std")
    assert not default.accepts("eval/episode_reward/alive")
    assert not default.accepts("train/loss")
    explicit = MetricFilter(manifest(metrics={"keys": ["eval/*", "eval/episode_reward/alive"]}).metrics)
    assert explicit.accepts("eval/episode_reward/alive") and not explicit.accepts(
        "eval/episode_reward/energy"
    )
