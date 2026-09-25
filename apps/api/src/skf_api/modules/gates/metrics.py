"""Resolve `<stage key>.<dot.path>` metrics against evaluation summaries (SPEC §7).

Keys may themselves contain dots (`stress.push_0.5mps.fall_rate`), so each level matches the longest
key that exists rather than splitting blindly on '.'.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9),
    "!=": lambda a, b: not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9),
}


def _lookup(node: Any, parts: Sequence[str]) -> Any:
    if not parts:
        return node
    if not isinstance(node, Mapping):
        return None
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if key in node:
            found = _lookup(node[key], parts[i:])
            if found is not None:
                return found
    return None


def resolve_metric(summaries: Mapping[str, Mapping[str, Any]], metric: str) -> float | None:
    """`summaries` maps stage key -> latest evaluation summary. Non-numeric values resolve to None."""
    value = _lookup(summaries, metric.split("."))
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def compare(actual: float | None, op: str, threshold: float) -> bool:
    return actual is not None and OPS[op](actual, threshold)
