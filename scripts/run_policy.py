"""Roll out a G1 policy headless, print gait metrics, optionally save a video.

    uv run scripts/run_policy.py --vx 0.5 --period 0.8 --video results/videos/baseline.mp4
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco

from g1pipe.sim import G1Sim, Perturb, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--vx", type=float, default=0.5)
ap.add_argument("--period", type=float, default=0.8, help="gait cycle length in seconds (trained value 0.8)")
ap.add_argument("--duration", type=float, default=10.0)
ap.add_argument("--policy", default=None, help="TorchScript policy; default = Unitree pre-trained")
ap.add_argument("--video", default=None)
ap.add_argument("--friction", type=float, default=1.0)
ap.add_argument("--payload", type=float, default=0.0)
ap.add_argument("--push-every", type=float, default=0.0)
ap.add_argument("--push-vel", type=float, default=0.0)
a = ap.parse_args()

sim = G1Sim(policy_path=a.policy)
renderer = mujoco.Renderer(sim.m, 480, 640) if a.video else None
p = Perturb(friction=a.friction, payload_kg=a.payload, push_every_s=a.push_every, push_vel=a.push_vel)
ep, frames = sim.run(cmd=(a.vx, 0, 0), period=a.period, duration=a.duration, perturb=p, renderer=renderer)
print(json.dumps(summarize(ep, a.vx, a.period), indent=2))
if a.video:
    Path(a.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.video, frames, fps=30)
    print("video:", a.video)
