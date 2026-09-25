"""Evaluation summary of a step-length run, from scripts/eval_suite.py's output directory.

    summary.json  grid_fall_rate, grid_step_abs_err_cm, grid_vx_abs_err, stress (list of cases), demo
    grid.csv      one row per (speed, step length, seed) of the E2 tracking grid

Stress cases are keyed by name so gate metrics can address them, e.g. `stress.push_0.5mps.fall_rate`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def summarize(out_dir: Path) -> dict[str, Any]:
    summary = json.loads((out_dir / "summary.json").read_text())
    stress = {
        row["case"]: {k: row.get(k) for k in ("fall_rate", "step_abs_err_cm", "vx_abs_err")}
        for row in summary.get("stress", [])
    }
    grid: list[dict[str, Any]] = []
    if (out_dir / "grid.csv").is_file():
        with open(out_dir / "grid.csv", newline="") as f:
            grid = [_typed(row) for row in csv.DictReader(f)]
    return {
        "grid_fall_rate": summary.get("grid_fall_rate"),
        "grid_step_abs_err_cm": summary.get("grid_step_abs_err_cm"),
        "grid_vx_abs_err": summary.get("grid_vx_abs_err"),
        "stress": stress,
        "demo": summary.get("demo"),
        "grid": grid,
    }


def _typed(row: dict[str, str]) -> dict[str, Any]:
    return {key: _value(cell) for key, cell in row.items()}


def _value(cell: str) -> Any:
    """CSV cell written by csv.DictWriter from Python values: bools, empty (None), ints, floats."""
    if cell in ("True", "False"):
        return cell == "True"
    if cell == "":
        return None
    for kind in (int, float):
        try:
            return kind(cell)
        except ValueError:
            pass
    return cell
