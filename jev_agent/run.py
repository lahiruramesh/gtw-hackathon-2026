"""Run the Jev-supervised G1 through perturbation scenarios, learn from each episode.

    python -m jev_agent.run runs/g1-steplength-v1/run/params.pkl --episodes 8
    python -m jev_agent.run PARAMS --brain offline          # dry run, no API calls
    python -m jev_agent.run PARAMS --baseline normal        # fixed-command comparison
    python -m jev_agent.run PARAMS --scenario pushes --mission "Carry a fragile box carefully." --video out.mp4

Writes runs/jev_agent/<tag>/{episodes.jsonl, decisions.jsonl, learner.json}.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from g1pipe.evaluate import Perturb, PolicyRunner
from jev_agent import questions as Q
from jev_agent.brain import make_brain
from jev_agent.learner import Learner
from jev_agent.perception import PPORobot
from jev_agent.supervisor import Supervisor

SCENARIOS = {
    "nominal": Perturb(),
    "slippery": Perturb(friction=0.35),
    "pushes": Perturb(push_every_s=3.0, push_vel=0.6),
    "payload": Perturb(payload_kg=6.0),
    "delay": Perturb(action_delay_steps=2),
    "storm": Perturb(friction=0.5, push_every_s=2.5, push_vel=0.5, payload_kg=3.0),
}
MISSIONS = [
    "Walk to the charging dock at the end of the hall. No rush.",
    "Hurry to the door, a person is waiting.",
    "Carry a fragile box across the room carefully.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--brain", choices=["jev", "offline"], default="jev")
    ap.add_argument("--baseline", choices=Q.MODE_ORDER, help="skip the agent; hold this gait mode")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--scenario", choices=list(SCENARIOS), help="default: cycle through all")
    ap.add_argument("--mission", help="default: cycle through MISSIONS")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-learn", action="store_true", help="use but don't update the learner")
    ap.add_argument("--video", default=None, help="record the last episode")
    ap.add_argument("--require-jev", action="store_true", help="fail instead of going offline without a key")
    a = ap.parse_args()

    tag = a.tag or (f"baseline_{a.baseline}" if a.baseline else a.brain)
    out = Path("runs/jev_agent") / tag
    out.mkdir(parents=True, exist_ok=True)
    runner = PolicyRunner(a.params)
    learner = Learner(out / "learner.json")
    brain = None if a.baseline else make_brain(a.brain, require_jev=a.require_jev)

    scen_names = [a.scenario] if a.scenario else list(SCENARIOS)
    missions = [a.mission] if a.mission else MISSIONS
    with open(out / "episodes.jsonl", "a") as ep_f, open(out / "decisions.jsonl", "a") as dec_f:
        for ep in range(a.episodes):
            scen, mission = scen_names[ep % len(scen_names)], missions[ep % len(missions)]
            renderer = None
            if a.video and ep == a.episodes - 1:
                import mujoco
                renderer = mujoco.Renderer(runner.m, 480, 640)
            if a.baseline:
                sup, schedule = None, (lambda t, m=a.baseline: Q.GAIT_MODES[m])
            else:
                sup = Supervisor(PPORobot(runner), brain, learner, mission, Q.GAIT_MODES)
                schedule = (lambda t, s=sup: Q.GAIT_MODES[s(t)])
            t0 = time.perf_counter()
            summary, frames, _, log = runner.run(duration=a.duration, perturb=SCENARIOS[scen],
                                                 renderer=renderer, seed=ep, schedule=schedule)
            wall = time.perf_counter() - t0
            dist = sum(v * runner.ctrl_dt for v in log["vx"])
            rec = {"episode": ep, "scenario": scen, "mission": mission, "agent": tag,
                   "fell": summary["fell"], "fall_time": summary["fall_time"],
                   "distance_m": round(dist, 2), "vx_abs_err": summary["vx_abs_err"],
                   "wall_s": round(wall, 1)}
            if sup:
                decs = sup.finish(summary["fall_time"])
                lat = [d["jev"]["latency_s"] for d in decs if not d["jev"]["error"]]
                rec.update(modes=dict(Counter(d["mode"] for d in decs)),
                           reasons=dict(Counter(d["reason"] for d in decs)),
                           brain_errors=sum(bool(d["jev"]["error"]) for d in decs),
                           jev_latency_ms=round(1000 * sum(lat) / len(lat), 1) if lat else None)
                for d in decs:
                    dec_f.write(json.dumps({"episode": ep, "scenario": scen, **d}) + "\n")
                if not a.no_learn:
                    learner.learn(decs)
                    learner.save()
                errs = [d["jev"]["error"] for d in decs if d["jev"]["error"]]
                if errs:
                    print("  brain error:", errs[0])
            ep_f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec))
            if frames:
                import imageio.v2 as imageio
                imageio.mimsave(a.video, frames, fps=30)
                print("video:", a.video)
    if not a.baseline and a.episodes:
        print("learned (n, mean reward) per context/mode:")
        for ctx, modes in learner.summary().items():
            print(f"  {ctx:28s}", {m: v for m, v in modes.items() if v[0]})
        print("fall-risk model:", "active" if learner.w is not None else "not enough falls yet")


if __name__ == "__main__":
    main()
