"""docs/effort_log.csv: `date,time,who,stage,hours,gpu_hours,what_changed,outcome,blocker`."""

from __future__ import annotations

import csv
from pathlib import Path


class EffortLog:
    def __init__(self, rows: list[dict[str, str]]):
        self._rows = rows

    @classmethod
    def read(cls, path: Path) -> EffortLog:
        if not path.is_file():
            return cls([])
        with path.open(newline="") as f:
            return cls(list(csv.DictReader(f)))

    def gpu_hours(self, match: str) -> float | None:
        """GPU hours of the first row whose description contains `match` and records them."""
        for row in self._rows:
            if match in (row.get("what_changed") or "") and (row.get("gpu_hours") or "").strip():
                try:
                    return float(row["gpu_hours"])
                except ValueError:
                    return None
        return None
