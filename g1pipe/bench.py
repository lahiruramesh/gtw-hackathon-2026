"""Fixed benchmark for comparing stairs policies across runs, batched in MJX on the training GPU.

g1pipe.gpu_eval ranks the checkpoints of one run under the conditions that run was trained with
(its delay mix, Playground's pushes). That cannot compare runs: v15 with 70 % delayed episodes would
be tested on a harder mix than v14. Here every policy meets the same conditions, the same start
states (one seed for all policies, so differences between policies are paired) and the same
disturbances, whatever it was trained with.

An episode follows the plain-MuJoCo strict test (g1pipe.stairs_eval.cross_cell): start on the
border of a held-out staircase of a random height, square to the steps, walk over the whole
staircase (up and down a pyramid, down and up a pit). It is crossed once the robot is past the far
border (CELL - BORDER from the start edge) without having fallen; falls after that (on the next
staircase) do not count. The conditions mirror scripts/certify.py, which stays the final gate:

  strict        0.7 m/s, true height scan, no delay, nominal friction, no payload, no pushes
  camera        + the head camera's scan errors
  delay_20ms    + one control step (20 ms) of actuation delay in every episode
  low_friction  + foot-floor friction x0.5
  payload_5kg   + 5 kg on the torso
  push          + a 0.3 m/s kick in a random direction every 3 s
  speed_06      0.6 m/s

Per condition and step height: episodes, crossed, fell. g1pipe.improve.stats turns these into fall
rates up to the target height, 95 % intervals and the distance to the targets (skills/stairs/summarize.py
for the web app's gate).

    python -m g1pipe.bench runs/g1-stairs-v14/run/params.pkl --out runs/g1-stairs-v14/run/bench.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize

from g1pipe import stairs_terrain as T
from g1pipe.gpu_eval import load_params, run_config, wilson

BENCH_VERSION = 2   # bump when a condition or the episode definition changes: cached results are then stale
CROSS_DIST = T.CELL - T.BORDER - 0.15   # m from the start (0.15 m inside the border): past the far border


@dataclasses.dataclass(frozen=True)
class Condition:
    vx: float = 0.7
    scan: str = "true"
    delay_steps: int = 0
    friction: float = 1.0      # multiplier on the nominal foot-floor friction
    payload_kg: float = 0.0
    push_every_s: float = 0.0
    push_vel: float = 0.0


CONDITIONS = {
    "strict": Condition(),
    "camera": Condition(scan="camera"),
    "delay_20ms": Condition(delay_steps=1),
    "low_friction": Condition(friction=0.5),
    "payload_5kg": Condition(payload_kg=5.0),
    "push": Condition(push_every_s=3.0, push_vel=0.3),
    "speed_06": Condition(vx=0.6),   # no dot: gate metrics are dot paths
}


def make_env(ref_path, c: Condition, impl=None):
    from g1pipe.stairs_env import STAIRS_CADENCE, StairsStepLength, default_config, floor_pairs
    cfg = default_config()
    cfg.impl = impl or ("warp" if jax.default_backend() == "gpu" else "jax")
    cfg.leg_action_scale = run_config(ref_path).get("leg_action_scale", 0.0)   # the network's action space
    cfg.action_delay_max = max(c.delay_steps, 1)
    cfg.action_delay_p = 1.0 if c.delay_steps else 0.0
    cfg.scan_model = "camera" if c.scan == "camera" else "uniform"
    cfg.push_config.enable = False   # pushes, if any, are the condition's own (see make_evaluator)
    cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_yaw = [c.vx, c.vx], [0.0, 0.0], [0.0, 0.0]
    cfg.gait_freq_range = [STAIRS_CADENCE, STAIRS_CADENCE]
    cfg.spawn_mode = "strict"
    env = StairsStepLength(config=cfg, layout="test", eval_levels=True)
    if c.friction != 1.0 or c.payload_kg:
        m = env._mjx_model
        floor = floor_pairs(m)
        pf = m.pair_friction.at[floor, 0:2].multiply(c.friction)
        mass = m.body_mass.at[g1_randomize.TORSO_BODY_ID].add(c.payload_kg)
        env._mjx_model = m.tree_replace({"pair_friction": pf, "body_mass": mass})
    return env, STAIRS_CADENCE


def make_evaluator(ref_path, c: Condition, episodes=2048, seconds=10.0, seed=0, impl=None):
    """One compiled batched rollout for condition c; returns f(params_path) -> result.
    Policies with the same network shape (every run warm-started from the same lineage) share it."""
    env, f = make_env(ref_path, c, impl)
    ref = load_params(ref_path)
    nets = ppo_networks.make_ppo_networks(ref["obs_size"], ref["action_size"],
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **ref["network_factory"])
    make_policy = ppo_networks.make_inference_fn(nets)
    cmd = jp.array([c.vx, 0.0])
    n_steps = int(seconds / env.dt)
    push_steps = int(round(c.push_every_s / env.dt)) if c.push_every_s > 0 else 0

    def pin(state):
        # Playground zeroes 10 % of commands and resamples every 500 steps; hold vx and the cadence
        info = dict(state.info, command=state.info["command"].at[:2].set(cmd),
                    gait_freq=jp.asarray(f), step_cmd=jp.asarray(c.vx / (2 * f)),
                    phase_dt=2 * jp.pi * env.dt * jp.array([f]))
        return state.replace(info=info)

    def rollout(params, key):
        policy = make_policy(params, deterministic=True)
        key, reset_key = jax.random.split(key)
        state = pin(env.reset(reset_key))

        def body(carry, i):
            state, fell, crossed = carry
            act, _ = policy(state.obs, jax.random.PRNGKey(0))
            state = pin(env.step(state, act))
            if push_steps:
                ang = jax.random.uniform(jax.random.fold_in(key, i), (), minval=0.0, maxval=2 * jp.pi)
                kick = jp.where((i + 1) % push_steps == 0, c.push_vel, 0.0) * jp.array([jp.cos(ang), jp.sin(ang)])
                state = state.replace(data=state.data.replace(qvel=state.data.qvel.at[:2].add(kick)))
            cur = state.info["curriculum"]
            fell = fell | (cur["fell"] & ~crossed)          # falls after the crossing do not count
            crossed = crossed | ((cur["dist"] > CROSS_DIST) & ~fell)
            return (state, fell, crossed), None

        init = (state, jp.zeros((), bool), jp.zeros((), bool))
        (state, fell, crossed), _ = jax.lax.scan(body, init, jp.arange(n_steps))
        return state.info["curriculum"]["level"], fell, crossed

    batched = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))
    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    rises = T.cell_rises("test")[:, 0]

    def run(path):
        level, fell, crossed = map(np.asarray, batched(load_params(path)["params"], keys))
        rows = []
        for lv in range(T.GRID):
            m = level == lv
            rows.append({"level": lv, "rise_cm": round(float(rises[lv]) * 100, 1), "n": int(m.sum()),
                         "crossed": int(crossed[m].sum()), "fell": int(fell[m].sum())})
        n, k = len(level), int(fell.sum())
        return {"episodes": n, "crossed": int(crossed.sum()), "fell": k, "fall_rate": k / n,
                "fall_ci95": list(wilson(k, n)), "per_level": rows}

    return run


def bench_key(episodes, seconds, seed):
    return f"v{BENCH_VERSION}-e{episodes}-s{seconds:g}-seed{seed}"


def run_bench(paths, conditions=None, episodes=2048, seconds=10.0, seed=0, impl=None, log=print):
    """{path: {"key": ..., "conditions": {name: result}}} for every policy under every condition.
    Condition-major: one compile per condition, shared by all policies."""
    names = list(conditions or CONDITIONS)
    out = {str(p): {"key": bench_key(episodes, seconds, seed), "conditions": {}} for p in paths}
    for name in names:
        t0 = time.time()
        run = make_evaluator(paths[0], CONDITIONS[name], episodes, seconds, seed, impl)
        for p in paths:
            r = run(p)
            out[str(p)]["conditions"][name] = r
            lo, hi = r["fall_ci95"]
            log(f"{name:13} {Path(p).parent.parent.name}/{Path(p).name}: fell {r['fell']}/{r['episodes']} "
                f"({r['fall_rate']*100:.1f} %, CI {lo*100:.1f}-{hi*100:.1f})")
        log(f"{name}: {time.time() - t0:.0f} s for {len(paths)} policies")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="+", help="params.pkl / ckpt_*.pkl files (same network shape)")
    ap.add_argument("--conditions", nargs="*", default=None, choices=list(CONDITIONS))
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", default=None, help="'warp' (NVIDIA GPU) or 'jax'")
    ap.add_argument("--out", required=True, help="JSON: {params path: {key, conditions: {name: result}}}")
    a = ap.parse_args()
    res = run_bench(a.params, a.conditions, a.episodes, a.seconds, a.seed, a.impl,
                    log=lambda m: print(m, flush=True))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
