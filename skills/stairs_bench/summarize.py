"""Summaries of a g1-stairs-bench run's evaluation stages (stdlib only; loaded by file path):

* bench: g1pipe.bench's `bench.json` (one policy, every condition, crossings and falls per step
  height). Per condition, the fall rate on step heights up to TARGET_CM (`rate`), its 95 % upper
  bound (`rate_hi`, what the gate checks) and the rate over every height (`rate_all`). `gap` is the
  sum over the gated conditions of how far `rate` is above its threshold (0 = on target): the number
  the improvement loop drives down (g1pipe.improve.stats.gap, same definition).
* evaluate: g1pipe.stairs_eval's `strict.json`, summarised exactly as the g1-stairs skill does.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

TARGET_CM = 10.0  # benchmark rates count step heights up to this (the release target)
# gated thresholds, as in skill.yaml's gate (kept in step by tests/test_improve.py)
THRESHOLDS = {
    "strict": 0.01,
    "camera": 0.02,
    "delay_20ms": 0.02,
    "low_friction": 0.05,
    "payload_5kg": 0.02,
    "push": 0.02,
    "speed_06": 0.02,
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % interval for a rate k/n (as g1pipe.gpu_eval.wilson)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def bench_condition(result: dict[str, Any], target_cm: float = TARGET_CM) -> dict[str, Any]:
    rows = result["per_level"]
    low = [r for r in rows if r["rise_cm"] <= target_cm + 1e-6]
    n, fell = sum(r["n"] for r in low), sum(r["fell"] for r in low)
    n_all, fell_all = sum(r["n"] for r in rows), sum(r["fell"] for r in rows)
    return {
        "n": n,
        "fell": fell,
        "rate": fell / n if n else None,
        "rate_hi": wilson(fell, n)[1],
        "n_all": n_all,
        "fell_all": fell_all,
        "rate_all": fell_all / n_all if n_all else None,
        "crossed_all": sum(r["crossed"] for r in rows),
        "by_height": [
            {"rise_cm": r["rise_cm"], "n": r["n"], "fell": r["fell"], "crossed": r["crossed"]} for r in rows
        ],
    }


def summarize_bench(bench: dict[str, Any]) -> dict[str, Any]:
    (entry,) = bench.values()  # one policy per stage
    out: dict[str, Any] = {name: bench_condition(r) for name, r in entry["conditions"].items()}
    out["gap"] = sum(
        max(0.0, out[k]["rate"] - thr)
        for k, thr in THRESHOLDS.items()
        if k in out and out[k]["rate"] is not None
    )
    out["bench_key"] = entry["key"]
    out["target_cm"] = TARGET_CM
    return out


def _stairs_summarizer():
    path = Path(__file__).resolve().parents[1] / "stairs" / "summarize.py"
    spec = importlib.util.spec_from_file_location("g1_stairs_summarize", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarize(out_dir: Path) -> dict[str, Any]:
    if (out_dir / "bench.json").is_file():
        return summarize_bench(json.loads((out_dir / "bench.json").read_text()))
    return _stairs_summarizer().summarize(out_dir)
