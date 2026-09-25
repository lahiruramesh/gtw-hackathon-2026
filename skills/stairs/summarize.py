"""Strict-test summary of a stairs run, from g1pipe.stairs_eval's `--out strict.json` (one entry per
crossing: rise_m, kind, fell, crossed, reached_centre, pelvis_rel_m [min, median], tilt_deg [median, max]).

The certified step height follows g1pipe.stairs_eval.certified_height exactly; it is re-implemented
here because importing g1pipe pulls in JAX.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Stability gate for one step height, as in g1pipe.stairs_eval: every staircase crossed, no falls,
# torso never tilted past GATE_TILT_DEG, pelvis never below GATE_PELVIS_M above the ground under it.
GATE_TILT_DEG = 25.0
GATE_PELVIS_M = 0.55


def stable(runs: list[dict[str, Any]]) -> bool:
    return all(
        x["crossed"]
        and not x["fell"]
        and x["tilt_deg"][1] < GATE_TILT_DEG
        and x["pelvis_rel_m"][0] > GATE_PELVIS_M
        for x in runs
    )


def certified_height(results: list[dict[str, Any]]) -> float:
    """Tallest step height (m) such that it and every lower height pass the stability gate; 0 if none."""
    by: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for x in results:
        by[x["rise_m"]].append(x)
    cert = 0.0
    for rise in sorted(by):
        if not stable(by[rise]):
            break
        cert = rise
    return cert


def summarize(out_dir: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = json.loads((out_dir / "strict.json").read_text())
    n = len(results)
    crossed = sum(bool(x["crossed"]) for x in results)
    fell = sum(bool(x["fell"]) for x in results)
    by: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for x in results:
        by[x["rise_m"]].append(x)
    return {
        "n": n,
        "crossed": crossed,
        "fell": fell,
        "crossed_rate": crossed / n if n else None,
        "fall_rate": fell / n if n else None,
        "certified_cm": round(certified_height(results) * 100, 2),
        "by_height": [_height_row(rise, by[rise]) for rise in sorted(by)],
    }


def _height_row(rise: float, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rise_cm": round(rise * 100, 2),
        "runs": len(runs),
        "crossed": sum(bool(x["crossed"]) for x in runs),
        "fell": sum(bool(x["fell"]) for x in runs),
        "reached_centre": sum(bool(x.get("reached_centre")) for x in runs),
        "max_tilt_deg": max(x["tilt_deg"][1] for x in runs),
        "min_pelvis_m": min(x["pelvis_rel_m"][0] for x in runs),
        "stable": stable(runs),
    }
