"""Distance to the targets and keep/drop decisions for improvement rounds. Pure Python, no JAX.

Inputs are benchmark summaries as skills/stairs/summarize.py writes them (per condition: n, fell,
rate, rate_hi on steps up to the target height) and the skill's gate criteria on them
(`bench.<condition>.rate_hi <= threshold`), so the loop's goal is exactly the web app's release gate.

* gap(summary): sum over conditions of how far the fall rate is above its threshold (0 = every
  condition at or under its threshold). The loop minimises it.
* passes(summary): every condition's 95 % upper bound is under its threshold (what the gate checks).
* compare(a, b): condition by condition, is a significantly better than b where b misses its
  target, and nowhere significantly worse, counting evaluation noise plus the measured
  seed-to-seed spread of training?
* seed_noise(a, b): two trainings of the same recipe with different seeds differ this much, on the
  gap and per condition. Trainings vary far more than 2048-episode evaluations do (the dry run and
  v10-v14 show it), so both tests add this spread to the binomial error.

Every decision here is arithmetic in code. Advisors (g1pipe.improve.advisor) only choose what to try next.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def targets_from_gate(gate: list[dict]) -> dict[str, float]:
    """{condition: threshold} from criteria like {metric: bench.delay_20ms.rate_hi, op: "<=", value: 0.02}."""
    out = {}
    for c in gate:
        parts = c["metric"].split(".")
        if len(parts) == 3 and parts[0] == "bench" and parts[2] == "rate_hi" and c["op"] in ("<=", "<"):
            out[parts[1]] = float(c["value"])
    return out


def _cond(summary: dict, name: str) -> dict:
    c = summary.get(name)
    if not c or not c.get("n"):
        raise KeyError(f"benchmark summary has no condition {name!r}")
    return c


def gap(summary: dict, targets: dict[str, float]) -> float:
    return sum(max(0.0, _cond(summary, k)["rate"] - thr) for k, thr in targets.items())


def gap_var(summary: dict, targets: dict[str, float]) -> float:
    """Binomial variance of gap(), counting every condition (conservative near the thresholds)."""
    v = 0.0
    for k in targets:
        c = _cond(summary, k)
        p = min(max(c["rate"], 0.5 / c["n"]), 1 - 0.5 / c["n"])   # never zero: 0/1280 is not certainty
        v += p * (1 - p) / c["n"]
    return v


def passes(summary: dict, targets: dict[str, float]) -> bool:
    return all(_cond(summary, k)["rate_hi"] <= thr for k, thr in targets.items())


def failing(summary: dict, targets: dict[str, float]) -> list[str]:
    return [k for k, thr in targets.items() if _cond(summary, k)["rate_hi"] > thr]


@dataclass
class Verdict:
    better: bool
    improvement: float          # gap(b) - gap(a), positive = a is closer to the targets
    z: float                    # the largest z among the improved conditions (0 if none)
    improved: list[str]         # off-target conditions where a is significantly better than b
    regressions: list[str]      # conditions where a is significantly worse than b
    reason: str
    z_by_condition: dict[str, float]


def compare(a: dict, b: dict, targets: dict[str, float], noise: dict | None = None,
            z_win: float = 2.5, z_guard: float = 2.5) -> Verdict:
    """Is policy a (summary) better than policy b?

    Per condition, z = (b's fall rate - a's) / sqrt(binomial error + seed noise of that condition).
    a is better when at least one condition that b has above its threshold improves at z >= z_win,
    no condition gets worse at z >= z_guard, and the gap to the targets shrinks. Testing conditions
    one by one keeps a large gain on one condition from drowning in the seed noise of another; with
    7 conditions, z 2.5 keeps the chance of a false win near 5 % per arm."""
    noise = noise or {"gap": 0.0, "cond": {}}
    zs, improved, regressions = {}, [], []
    for k, thr in targets.items():
        ca, cb = _cond(a, k), _cond(b, k)
        pa, pb = ca["fell"] / ca["n"], cb["fell"] / cb["n"]
        pool = (ca["fell"] + cb["fell"]) / (ca["n"] + cb["n"])
        s = math.sqrt(max(pool * (1 - pool), 0.25 / (ca["n"] + cb["n"])) * (1 / ca["n"] + 1 / cb["n"])
                      + noise["cond"].get(k, 0.0) ** 2)
        z = (pb - pa) / s
        zs[k] = round(z, 2)
        if z >= z_win and pb > thr:
            improved.append(k)
        if -z >= z_guard:
            regressions.append(k)
    d = gap(b, targets) - gap(a, targets)
    better = bool(improved) and not regressions and d > 0
    if regressions:
        reason = f"worse on {', '.join(f'{k} (z {-zs[k]:.1f})' for k in regressions)}"
    elif not improved:
        best = max(zs, key=zs.get)
        reason = f"no condition off target improved beyond noise (best: {best}, z {zs[best]:.1f} < {z_win})"
    elif d <= 0:
        reason = f"better on {', '.join(improved)} but no closer to the targets overall (gap change {-d:+.4f})"
    else:
        reason = f"better on {', '.join(f'{k} (z {zs[k]:.1f})' for k in improved)}; gap {-d:+.4f}"
    return Verdict(better=better, improvement=d, z=max((zs[k] for k in improved), default=0.0),
                   improved=improved, regressions=regressions, reason=reason, z_by_condition=zs)


def seed_noise(a: dict, b: dict, targets: dict[str, float]) -> dict:
    """How far two trainings of the same recipe (different seeds) are apart: on the gap and per
    condition. One pair is a rough estimate; campaign.py averages every pair it has."""
    return {"gap": abs(gap(a, targets) - gap(b, targets)),
            "cond": {k: abs(_cond(a, k)["rate"] - _cond(b, k)["rate"]) for k in targets}}


def pool_noise(samples: list[dict]) -> dict | None:
    """Root-mean-square of several seed_noise() samples."""
    if not samples:
        return None
    rms = lambda xs: math.sqrt(sum(x * x for x in xs) / len(xs))
    keys = samples[0]["cond"].keys()
    return {"gap": rms([s["gap"] for s in samples]), "cond": {k: rms([s["cond"][k] for s in samples]) for k in keys},
            "samples": len(samples)}


def describe(summary: dict, targets: dict[str, float]) -> dict[str, str]:
    """Words for each condition, for advisors that should not do arithmetic (Jev)."""
    out = {}
    for k, thr in targets.items():
        c = _cond(summary, k)
        r = c["rate"]
        if c["rate_hi"] <= thr:
            out[k] = "meets the target"
        elif r <= thr:
            out[k] = "about at the target, not yet proven"
        elif r <= 3 * thr:
            out[k] = "a little above the target"
        elif r <= 10 * thr:
            out[k] = "well above the target"
        else:
            out[k] = "far above the target"
    return out
