"""Certify a stairs policy in plain MuJoCo and write its policy card.

Runs the certification suite (held-out stairs, several start poses, the head camera, other speeds,
tread depths and disturbances) as parallel g1pipe.stairs_eval processes, then writes
<run>/card_<ckpt>.json and .md: lineage, per-condition crossing and fall rates with 95 %
intervals, certified step height per condition, robot-safety numbers, and the certified envelope
(the step heights and conditions the supervisor may send it into).

    uv run python scripts/certify.py runs/g1-stairs-v13/run/params.pkl --target-cm 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1pipe.gpu_eval import wilson  # noqa: E402

# name: (stairs_eval args, required for the envelope)
SUITE = {
    "strict":        (["--vx", "0.7", "--starts", "3"], True),
    "strict_camera": (["--vx", "0.7", "--starts", "3", "--scan", "camera"], True),
    "speed_0.6":     (["--vx", "0.6", "--starts", "3"], False),
    "tread_30cm":    (["--vx", "0.7", "--layout", "train30"], False),
    "push":          (["--vx", "0.7", "--perturb", "push"], False),
    "low_friction":  (["--vx", "0.7", "--perturb", "lowfric"], False),
    "payload_5kg":   (["--vx", "0.7", "--perturb", "payload"], False),
    "delay_20ms":    (["--vx", "0.7", "--perturb", "delay"], False),
}


def run_one(params, name, args, out_dir):
    out = out_dir / f"{name}.json"
    if not out.exists():
        cmd = [sys.executable, "-m", "g1pipe.stairs_eval", str(params), *args, "--out", str(out)]
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return name, json.loads(out.read_text())


def pct(x):
    return "n/a" if x is None else f"{x*100:.1f} %"


def certified(runs):
    """Tallest step height such that it and every lower one: all crossed, none fell."""
    by = {}
    for x in runs:
        by.setdefault(x["rise_m"], []).append(x)
    best = 0.0
    for h in sorted(by):
        if all(x["crossed"] and not x["fell"] for x in by[h]):
            best = h
        else:
            break
    return best


def condition(runs, target_m):
    n = len(runs)
    fell = sum(x["fell"] for x in runs)
    low = [x for x in runs if x["rise_m"] <= target_m + 1e-6]
    fell_low = sum(x["fell"] for x in low)
    return {
        "runs": n, "crossed": sum(x["crossed"] for x in runs), "fell": fell,
        "fall_rate": fell / n, "fall_ci95": wilson(fell, n),
        "runs_to_target": len(low), "fell_to_target": fell_low, "fall_ci95_to_target": wilson(fell_low, len(low)),
        "certified_m": certified(runs),
        # None for results recorded before the safety numbers existed
        "torque_sat_frac_median": float(np.median([x["torque_sat_frac"] for x in runs])) if "torque_sat_frac" in runs[0] else None,
        "joint_limit_frac_median": float(np.median([x["joint_limit_frac"] for x in runs])) if "joint_limit_frac" in runs[0] else None,
        "per_height": {f"{h*100:.1f}": f"{sum(x['crossed'] for x in xs)}/{len(xs)} fell {sum(x['fell'] for x in xs)}"
                       for h, xs in sorted({r["rise_m"]: [y for y in runs if y["rise_m"] == r["rise_m"]] for r in runs}.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--target-cm", type=float, default=10.0, help="step height the policy must handle")
    ap.add_argument("--only", nargs="*", default=None, choices=list(SUITE))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    params = Path(a.params).resolve()
    tag = params.stem
    out_dir = params.parent / f"cert_{tag}"
    out_dir.mkdir(exist_ok=True)
    suite = {k: v for k, v in SUITE.items() if a.only is None or k in a.only}

    results = {}
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_one, params, k, args, out_dir) for k, (args, _) in suite.items()]
        for f in cf.as_completed(futs):
            name, runs = f.result()
            results[name] = runs
            print(f"  done: {name}", flush=True)

    target = a.target_cm / 100
    conds = {k: condition(v, target) for k, v in results.items()}
    required = [k for k, (_, req) in suite.items() if req and k in conds]
    envelope_m = min(conds[k]["certified_m"] for k in required) if required else 0.0
    # pass: every required condition is clean (all crossed, no falls) up to the tallest tested
    # step at or below the target
    heights = sorted({x["rise_m"] for runs in results.values() for x in runs})
    needed = max([h for h in heights if h <= target + 1e-6], default=0.0)
    passed = bool(required) and all(conds[k]["certified_m"] >= needed - 1e-6 for k in required)
    run_cfg = params.with_name("config.json")
    lineage = {}
    if run_cfg.exists():
        c = json.loads(run_cfg.read_text())
        lineage = {"init_from": c.get("init_from"), "timesteps": c["ppo"]["num_timesteps"],
                   "learning_rate": c["ppo"]["learning_rate"],
                   **{k: c["env"].get(k) for k in ("leg_action_scale", "gait_freq_range", "scan_model")}}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    card = {"policy": str(params.relative_to(ROOT)), "git": commit, "lineage": lineage,
            "target_cm": a.target_cm, "envelope_cm": round(envelope_m * 100, 1), "pass": bool(passed),
            "required": required, "conditions": conds}
    (params.parent / f"card_{tag}.json").write_text(json.dumps(card, indent=1))

    md = [f"# Policy card: `{card['policy']}`", "",
          f"git `{commit}` · trained from `{lineage.get('init_from')}` · {lineage.get('timesteps', 0)/1e6:.0f}M steps · "
          f"lr {lineage.get('learning_rate')} · scan `{lineage.get('scan_model')}`", "",
          f"**Certified envelope: stairs ≤ {card['envelope_cm']} cm** (0.7 m/s, 1.4 Hz cadence; required: "
          f"{', '.join(required)}) · target {a.target_cm} cm: **{'PASS' if passed else 'NOT YET'}**", "",
          "| condition | runs | crossed | fell | fall rate (95 % CI) | certified | torque-saturated | joint at limit |",
          "|---|---|---|---|---|---|---|---|"]
    for k in suite:
        if k not in conds:
            continue
        c = conds[k]
        lo, hi = c["fall_ci95"]
        md.append(f"| {k}{' (required)' if k in required else ''} | {c['runs']} | {c['crossed']} | {c['fell']} | "
                  f"{c['fall_rate']*100:.1f} % ({lo*100:.1f}-{hi*100:.1f}) | {c['certified_m']*100:.1f} cm | "
                  f"{pct(c['torque_sat_frac_median'])} | {pct(c['joint_limit_frac_median'])} |")
    md += ["", "Per step height (crossed/runs, falls):", ""]
    for k in suite:
        if k in conds:
            md.append(f"- **{k}**: " + ", ".join(f"{h} cm {v}" for h, v in conds[k]["per_height"].items()))
    (params.parent / f"card_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
