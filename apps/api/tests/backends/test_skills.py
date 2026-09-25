"""Skill manifests and summarizers (skills/*), checked against real evaluation results in results/."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from backend_fakes import REPO_ROOT

SKILLS = REPO_ROOT / "skills"
RESULTS = REPO_ROOT / "results"
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_summarizer(skill: str) -> Callable[[Path], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"summarize_{skill}", SKILLS / skill / "summarize.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


@pytest.mark.parametrize("manifest", sorted(SKILLS.glob("*/skill.yaml")), ids=lambda p: p.parent.name)
def test_manifest_placeholders_and_flags_refer_to_params(manifest: Path) -> None:
    data = yaml.safe_load(manifest.read_text())
    params = set(data["params"])
    for stage in data["pipeline"]:
        for token in stage.get("argv", []):
            for name in PLACEHOLDER.findall(token):
                assert name in params | {"out_dir", "input_dir"}, f"{stage['id']}: unknown {{{name}}}"
        for param in stage.get("flags", {}):
            assert param in params, f"{stage['id']}: flag for unknown param {param}"
    for preset in data.get("presets", []):
        assert set(preset["params"]) <= params, f"preset {preset['id']} sets unknown params"
    stage_ids = {s["id"] for s in data["pipeline"]}
    for criterion in [*data["gate"], *data["headline"]]:
        assert criterion["metric"].split(".", 1)[0] in stage_ids


def test_both_skills_are_present() -> None:
    ids = {yaml.safe_load(p.read_text())["id"] for p in SKILLS.glob("*/skill.yaml")}
    assert {"g1-step-length", "g1-stairs"} <= ids


def test_step_length_summary_of_v1(tmp_path: Path) -> None:
    summary = load_summarizer("step_length")(RESULTS / "eval_v1")
    source = json.loads((RESULTS / "eval_v1" / "summary.json").read_text())
    assert summary["grid_fall_rate"] == 0.0
    assert summary["grid_step_abs_err_cm"] == pytest.approx(10.6826, abs=1e-3)
    assert summary["grid_vx_abs_err"] == source["grid_vx_abs_err"]
    assert set(summary["stress"]) == {row["case"] for row in source["stress"]}
    assert summary["stress"]["push_0.5mps"]["fall_rate"] == 0.0
    assert summary["stress"]["latency_20ms"]["fall_rate"] == 1.0
    assert summary["demo"]["n_steps"] == 36
    assert len(summary["grid"]) == 40
    first = summary["grid"][0]
    assert first["vx"] == 0.3 and first["seed"] == 0 and first["fell"] is False and first["fall_time"] is None
    json.dumps(summary, allow_nan=False)


def test_step_length_summary_without_grid_csv(tmp_path: Path) -> None:
    shutil.copy(RESULTS / "eval_nodr" / "summary.json", tmp_path / "summary.json")
    summary = load_summarizer("step_length")(tmp_path)
    assert summary["grid"] == []
    assert "nominal" in summary["stress"]


def stairs_summary(name: str, tmp_path: Path) -> dict[str, Any]:
    shutil.copy(RESULTS / "stairs" / name, tmp_path / "strict.json")
    return load_summarizer("stairs")(tmp_path)


@pytest.mark.parametrize(
    ("name", "crossed", "fell", "certified_cm"),
    [
        # crossed/fell as recorded in docs/effort_log.csv for these 96-run strict tests
        ("strict_v11_ckpt_00190709760.json", 88, 8, 8.14),
        ("strict_v10_ckpt_00253624320.json", 82, 14, 8.14),
        ("strict_v9_ckpt_00369295360.json", 81, 9, 4.71),
    ],
)
def test_stairs_summary_matches_the_effort_log(
    name: str, crossed: int, fell: int, certified_cm: float, tmp_path: Path
) -> None:
    summary = stairs_summary(name, tmp_path)
    assert (summary["n"], summary["crossed"], summary["fell"]) == (96, crossed, fell)
    assert summary["crossed_rate"] == pytest.approx(crossed / 96)
    assert summary["fall_rate"] == pytest.approx(fell / 96)
    assert summary["certified_cm"] == certified_cm
    rows = summary["by_height"]
    assert [r["rise_cm"] for r in rows] == sorted(r["rise_cm"] for r in rows)
    assert sum(r["runs"] for r in rows) == 96 and sum(r["fell"] for r in rows) == fell


def test_certified_height_is_the_last_stable_height_in_a_row(tmp_path: Path) -> None:
    summary = stairs_summary("strict_v11_ckpt_00190709760.json", tmp_path)
    stable = [r["stable"] for r in summary["by_height"]]
    certified_rows = [r for r in summary["by_height"] if r["rise_cm"] <= summary["certified_cm"]]
    assert all(r["stable"] for r in certified_rows)
    assert stable[len(certified_rows)] is False


def test_certified_height_rule(tmp_path: Path) -> None:
    def run(rise: float, *, fell: bool = False, tilt: float = 10.0, pelvis: float = 0.7) -> dict[str, Any]:
        return {
            "rise_m": rise,
            "crossed": not fell,
            "fell": fell,
            "reached_centre": True,
            "tilt_deg": [5.0, tilt],
            "pelvis_rel_m": [pelvis, 0.75],
        }

    cases = {
        "all stable": ([run(0.03), run(0.05)], 5.0),
        "tilt at the limit fails": ([run(0.03), run(0.05, tilt=25.0)], 3.0),
        "low pelvis fails": ([run(0.03, pelvis=0.55)], 0.0),
        "a fall stops the ladder": ([run(0.03), run(0.05, fell=True), run(0.07)], 3.0),
    }
    for label, (runs, expected) in cases.items():
        (tmp_path / "strict.json").write_text(json.dumps(runs))
        assert load_summarizer("stairs")(tmp_path)["certified_cm"] == expected, label
