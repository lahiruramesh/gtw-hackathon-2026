"""Measure each robot's normal-walking sensor levels; perception describes readings relative to them.

    python -m jev_agent.calibrate
"""
import numpy as np, json
from g1pipe.evaluate import PolicyRunner
from g1pipe.sim import G1Sim, yaw_of
from jev_agent.perception import PPORobot, VendorRobot, Perception, describe
from jev_agent import questions as Q
from jev_agent.stairs import SUP_MODES, VENDOR_MODES
KEYS = ["max_tilt_deg", "max_gyro", "max_slip", "vx_err", "lat_speed", "step_err"]
def collect(kind):
    rows = []
    if kind == "v1":
        r = PolicyRunner("runs/g1-steplength-v1/run/params.pkl"); rob = PPORobot(r)
        for mode in ("cautious", "normal", "stride"):
            per = Perception(rob); vx, st = Q.GAIT_MODES[mode]
            def sched(t):
                per.update(t, vx, st) if t > 0 else None
                if t > 2 and int(round(t / 0.02)) % 25 == 0:
                    rows.append({"mode": mode, **per.snapshot()}); per.end_window()
                return (vx, st)
            r.run(duration=14, schedule=sched, seed=1)
    else:
        for mode in ("cautious", "normal", "stride"):
            sim = G1Sim(); rob = VendorRobot(sim); per = Perception(rob); vx, T = VENDOR_MODES[mode]
            def cmd(t):
                per.update(t, vx, vx * T / 2)
                if t > 2 and int(round(t / 0.02)) % 25 == 0:
                    rows.append({"mode": mode, **per.snapshot()}); per.end_window()
                return (vx, 0, float(np.clip(-1.5 * yaw_of(sim.d.qpos[3:7]), -.5, .5)))
            sim.run(duration=14, cmd_fn=cmd, period=T)
    return rows
out = {}
for kind in ("v1", "vendor"):
    rows = collect(kind)
    out[kind] = {k: [round(float(np.percentile([r[k] for r in rows], p)), 3) for p in (50, 90, 99)] for k in KEYS}
    words = {}
    for r in rows:
        for k, v in describe(r).items(): words.setdefault(k, {}).setdefault(v, 0); words[k][v] += 1
    print(f"\n== {kind}: nominal walking, {len(rows)} windows (p50/p90/p99)"); [print(f"  {k:13s} {v}") for k, v in out[kind].items()]
    print("  current words:", {k: v for k, v in words.items() if k not in ('recent_push',)})
from pathlib import Path
note = ("p90 of each feature over 0.5 s windows of normal walking on flat nominal ground "
        "(cautious/normal/stride gaits, 12 s each). Regenerate: python -m jev_agent.calibrate")
Path(__file__).with_name("calibration.json").write_text(json.dumps(
    {"_note": note, **{k: {f: v[1] for f, v in d.items()} for k, d in out.items()}}, indent=1))
print("wrote jev_agent/calibration.json")
