"""Summary of a ring-pick-drop evaluation stage, from g1pipe.ring_eval's `--out eval.json` (a list with
one row per params file, best first; the stage evaluates one). Stdlib only; loaded by file path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize(out_dir: Path) -> dict[str, Any]:
    row = json.loads((out_dir / "eval.json").read_text())[0]
    out = {
        k: row[k]
        for k in (
            "episodes",
            "success",
            "success_rate",
            "lifted",
            "placed",
            "raceway",
            "dropped",
            "xy_err_median_mm",
        )
    }
    out["success_ci_lo"], out["success_ci_hi"] = row["success_ci95"]
    return out
