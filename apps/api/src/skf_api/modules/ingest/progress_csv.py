"""Parse g1pipe.train's progress.csv (`step,wall_s,<metric>...`) into metric points."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from skf_api.modules.ingest.service import Point


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_progress_csv(path: Path) -> list[Point]:
    points = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = _number(row.pop("step", "") or "")
            if step is None:
                continue
            wall_s = _number(row.pop("wall_s", "") or "")
            values = {k: v for k, raw in row.items() if k and (v := _number(raw or "")) is not None}
            points.append(Point(step=int(step), values=values, wall_s=wall_s))
    return points
