"""Release gate metric resolution and verdicts (SPEC §8)."""

from __future__ import annotations

from typing import Any

import pytest

from skf_api.modules.gates.metrics import compare, resolve_metric
from skf_api.modules.gates.models import GateVerdict
from skf_api.modules.gates.service import check_criteria, describe, level_results, simulation_verdict
from skf_api.settings import REPO_ROOT
from skf_api.skills_registry.loader import load_skills

SUMMARIES: dict[str, dict[str, Any]] = {
    "evaluate": {
        "grid_fall_rate": 0.0,
        "grid_step_abs_err_cm": 2.4,
        "stress": {"push_0.5mps": {"fall_rate": 0.0}, "latency_20ms": {"fall_rate": 1.0}},
        "demo": {"fell": False},
        "label": "text",
    },
}


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("evaluate.grid_step_abs_err_cm", 2.4),
        ("evaluate.stress.push_0.5mps.fall_rate", 0.0),  # a key with a dot in it
        ("evaluate.stress.latency_20ms.fall_rate", 1.0),
        ("evaluate.demo.fell", 0.0),  # booleans count as 0/1
        ("evaluate.label", None),  # not numeric
        ("evaluate.missing", None),
        ("strict.crossed_rate", None),  # no evaluation for that stage key
    ],
)
def test_resolve_metric(metric: str, expected: float | None) -> None:
    assert resolve_metric(SUMMARIES, metric) == expected


@pytest.mark.parametrize(
    ("actual", "op", "value", "passed"),
    [
        (0.0, "==", 0, True),
        (1e-12, "==", 0, True),
        (0.1, "==", 0, False),
        (0.1, "!=", 0, True),
        (2.9, "<", 3, True),
        (3.0, "<", 3, False),
        (3.0, "<=", 3, True),
        (0.9, ">=", 0.9, True),
        (0.89, ">", 0.9, False),
        (None, ">=", 0, False),
    ],
)
def test_compare(actual: float | None, op: str, value: float, passed: bool) -> None:
    assert compare(actual, op, value) is passed


def test_step_length_gate_on_summaries() -> None:
    manifest = load_skills(REPO_ROOT / "skills").skills["g1-step-length"].manifest
    verdict, criteria = check_criteria(manifest, SUMMARIES)
    assert verdict is GateVerdict.FAIL
    assert [c["passed"] for c in criteria] == [True, True, True, False]
    assert criteria[3] == {
        "metric": "evaluate.stress.latency_20ms.fall_rate",
        "label": "Survives 20 ms latency",
        "op": "==",
        "value": 0.0,
        "actual": 1.0,
        "passed": False,
        "level": "release",
    }
    assert [c["level"] for c in criteria] == ["simulation", "simulation", "release", "release"]

    passing = {
        "evaluate": {
            **SUMMARIES["evaluate"],
            "stress": {"push_0.5mps": {"fall_rate": 0.0}, "latency_20ms": {"fall_rate": 0.0}},
        }
    }
    assert check_criteria(manifest, passing)[0] is GateVerdict.PASS
    assert check_criteria(manifest, {})[0] is GateVerdict.FAIL  # missing evaluation fails


def test_levels_split_simulation_from_release() -> None:
    """Simulation criteria gate on their own; the release gate needs every criterion."""
    manifest = load_skills(REPO_ROOT / "skills").skills["g1-step-length"].manifest
    _, criteria = check_criteria(manifest, SUMMARIES)
    assert level_results(criteria) == [
        {"level": "simulation", "verdict": GateVerdict.PASS, "passed": 2, "total": 2},
        {"level": "release", "verdict": GateVerdict.FAIL, "passed": 3, "total": 4},
    ]
    assert simulation_verdict(criteria) is GateVerdict.PASS
    assert describe(criteria) == "simulation pass (2/2), release fail (3/4)"

    falls = {"evaluate": {**SUMMARIES["evaluate"], "grid_fall_rate": 0.5}}
    assert simulation_verdict(check_criteria(manifest, falls)[1]) is GateVerdict.FAIL


def test_decisions_stored_before_levels_count_as_release_only() -> None:
    legacy = [{"metric": "m", "label": "m", "op": "<", "value": 1.0, "actual": 0.5, "passed": True}]
    assert level_results(legacy) == [
        {"level": "release", "verdict": GateVerdict.PASS, "passed": 1, "total": 1}
    ]
    assert simulation_verdict(legacy) is None


def test_stairs_simulation_gate_is_crossing_and_falls() -> None:
    manifest = load_skills(REPO_ROOT / "skills").skills["g1-stairs"].manifest
    v14 = {"evaluate": {"crossed_rate": 95 / 96, "fell": 1, "certified_cm": 9.86}}
    _, criteria = check_criteria(manifest, v14)
    assert describe(criteria) == "simulation pass (2/2), release fail (2/3)"
