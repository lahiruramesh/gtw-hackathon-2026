"""Release gate metric resolution and verdicts (SPEC §8)."""

from __future__ import annotations

from typing import Any

import pytest

from skf_api.modules.gates.metrics import compare, resolve_metric
from skf_api.modules.gates.models import GateVerdict
from skf_api.modules.gates.service import check_criteria
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
    }

    passing = {
        "evaluate": {
            **SUMMARIES["evaluate"],
            "stress": {"push_0.5mps": {"fall_rate": 0.0}, "latency_20ms": {"fall_rate": 0.0}},
        }
    }
    assert check_criteria(manifest, passing)[0] is GateVerdict.PASS
    assert check_criteria(manifest, {})[0] is GateVerdict.FAIL  # missing evaluation fails
