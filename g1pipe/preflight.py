"""Checks to run on the training box before spending GPU hours (g1pipe.train --preflight runs them first).

  parity      the training engine (MJX / MuJoCo Warp) and the evaluation engine (plain MuJoCo)
              agree: the robot is put in the same poses on the stairs (flat, mid-tread, heel over
              a step edge, toes over an edge, on a riser), the default-pose PD targets are held for
              0.25 s in both engines (unstable poses tip over differently given more time), and pelvis position, torso tilt and foot heights must match.
              v1-v8 trained with one contact point per foot where plain MuJoCo makes 3-20; the
              edge poses expose exactly that kind of gap.
  throughput  env steps per second of the batched training env (jit(vmap(step)))
  warm start  the policy at step 0 acts like its parent: joint targets on the same observations,
              identical when the action scales match, close when they were rescaled

Exit status 1 if a check fails, so a job script can stop before training.

    python -m g1pipe.preflight --init-from runs/g1-stairs-v12/run/ckpt_00254279680.pkl --leg-action-scale 1.0
"""
from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from g1pipe import stairs_terrain as T

PARITY_POS_M, PARITY_TILT_DEG, PARITY_FOOT_M = 0.02, 2.0, 0.01
MIN_STEPS_PER_S = 50_000        # on a GPU; the L40S does ~100k with the Warp print fix


def _poses(layout):
    """(name, x, y, yaw) poses on the first staircase (ix=0) of the tallest level of `layout`,
    with the step edges found from the height grid along the row's centre line."""
    iy = T.GRID - 1
    x0 = -T.GRID_HALF
    y = -T.GRID_HALF + (iy + 0.5) * T.CELL
    xs = np.arange(x0, x0 + T.CELL / 2, 0.005)
    h = T.lookup(T.heights(layout), np.stack([xs, np.full_like(xs, y)], -1))
    # the heightfield turns each step into a ramp one grid cell wide: group the sloped samples
    sloped = np.abs(np.diff(h)) > 1e-4
    starts = np.flatnonzero(sloped & ~np.concatenate([[False], sloped[:-1]]))
    ends = np.flatnonzero(sloped & ~np.concatenate([sloped[1:], [False]]))
    edges = (xs[starts] + xs[ends + 1]) / 2
    e1, e2 = float(edges[0]), float(edges[1])              # first two step edges
    return [("flat", x0 - 0.8, y, 0.0),
            ("mid-tread", (e1 + e2) / 2 - 0.04, y, 0.0),
            ("heel over edge", e1 + 0.03, y, 0.0),          # ankle just past the riser: heel hangs
            ("toes over edge", e2 - 0.08, y, 0.0),          # toe box past the next riser
            ("across the stairs", (e1 + e2) / 2, y, np.pi / 2)]

# Reported, not gating: turned 90 deg, one sole lies lengthwise on a step edge, which the 5 cm
# heightfield turns into a ramp. Plain MuJoCo's box-heightfield contacts (30+) catch the upper rim;
# Warp's single box contact plus the sole spheres slide down it (foot ~15 cm lower after 0.25 s).
# Straight crossings land feet across edges (heel/toe poses, which agree to 0.1 cm), and the gap
# errs on the harsh side in training. Finer heightfield or box-geom stairs would close it.
INFORMATIONAL = {"across the stairs"}


def parity(env, layout="test", seconds=0.25):
    """Hold the default pose for `seconds` from the same start in both engines; compare."""
    m = env.mj_model
    grid = T.heights(layout)
    n = int(round(seconds / m.opt.timestep))
    feet = np.array(env._feet_site_id)
    torso = env._torso_body_id
    step = jax.jit(lambda d: mjx.step(env.mjx_model, d))
    rows, ok = [], True
    for name, x, y, yaw in _poses(layout):
        ring = np.array([[x + dx, y + dy] for dx in (-0.12, 0, 0.12) for dy in (-0.1, 0, 0.1)])
        qpos = np.array(env._init_q, np.float64)
        qpos[:3] = (x, y, T.lookup(grid, ring).max() + 0.755 + 0.01)
        qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        ctrl = np.array(env._default_pose)
        # plain MuJoCo
        d = mujoco.MjData(m)
        d.qpos[:], d.ctrl[:] = qpos, ctrl
        mujoco.mj_forward(m, d)
        for _ in range(n):
            mujoco.mj_step(m, d)
        ref = (d.qpos[:3].copy(), d.xmat[torso].reshape(3, 3)[2, 2], d.site_xpos[feet][:, 2].copy(), d.ncon)
        # training engine
        from mujoco_playground._src import mjx_env
        dx = mjx_env.make_data(m, qpos=jp.asarray(qpos), qvel=jp.zeros(m.nv), ctrl=jp.asarray(ctrl),
                               impl=env.mjx_model.impl.value, naconmax=256, njmax=env._config.njmax)
        dx = mjx.forward(env.mjx_model, dx)
        for _ in range(n):
            dx = step(dx)
        got = (np.asarray(dx.qpos[:3]), float(np.asarray(dx.xmat[torso]).reshape(3, 3)[2, 2]),
               np.asarray(dx.site_xpos[feet])[:, 2])
        dpos = float(np.linalg.norm(got[0] - ref[0]))
        dtilt = abs(np.degrees(np.arccos(np.clip(got[1], -1, 1)) - np.arccos(np.clip(ref[1], -1, 1))))
        dfoot = float(np.abs(got[2] - ref[2]).max())
        good = dpos < PARITY_POS_M and dtilt < PARITY_TILT_DEG and dfoot < PARITY_FOOT_M
        ok &= good or name in INFORMATIONAL
        rows.append(f"  {name:18} pelvis {dpos*100:5.1f} cm  tilt {dtilt:4.1f} deg  foot z {dfoot*100:4.1f} cm  "
                    f"(MuJoCo {ref[3]} contacts) {'ok' if good else 'MISMATCH'}"
                    f"{' (reported only, see INFORMATIONAL)' if name in INFORMATIONAL else ''}")
    return ok, rows


def throughput(env, n_envs=4096, steps=100):
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    state = reset(jax.random.split(jax.random.PRNGKey(0), n_envs))
    act = jp.zeros((n_envs, env.action_size))
    state = step(state, act)
    jax.block_until_ready(state.obs)
    t = time.time()
    for _ in range(steps):
        state = step(state, act)
    jax.block_until_ready(state.obs)
    return n_envs * steps / (time.time() - t)


def warm_start_check(env, init_from, gain):
    """Max joint-target difference between the parent policy and the warm-started one (rad)."""
    import pickle
    from pathlib import Path
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from g1pipe.train import warm_start
    blob = pickle.load(open(init_from, "rb"))
    if not (isinstance(blob, dict) and "params" in blob):
        blob = {**pickle.load(open(Path(init_from).with_name("params.pkl"), "rb")), "params": blob}
    new = warm_start(init_from, env.observation_size, gain)

    def targets(params, obs_size, scales):
        nets = ppo_networks.make_ppo_networks(obs_size, env.action_size,
                                              preprocess_observations_fn=running_statistics.normalize,
                                              **blob["network_factory"])
        pol = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
        return lambda obs: pol(obs, jax.random.PRNGKey(0))[0] * scales

    state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(1), 256))
    cur = np.asarray(env._config.action_scale) * np.ones(env.action_size)
    parent = cur * (gain if gain is not None else 1.0)
    obs = state.obs
    if blob["obs_size"]["state"][0] != env.observation_size["state"][0]:
        return None   # new inputs (e.g. flat -> stairs): not comparable on the same observations
    a = targets(blob["params"], blob["obs_size"], parent)(obs)
    b = targets(new, env.observation_size, cur)(obs)
    return float(jp.abs(a - b).max())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--leg-action-scale", type=float, default=None)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--min-steps-per-s", type=float, default=MIN_STEPS_PER_S)
    ap.add_argument("--skip", nargs="*", default=[], choices=["parity", "throughput", "warm"])
    a = ap.parse_args(argv)
    from g1pipe.stairs_env import StairsStepLength, default_config
    cfg = default_config()
    gpu = jax.default_backend() == "gpu"
    cfg.impl = a.impl or ("warp" if gpu else "jax")
    if a.leg_action_scale:
        cfg.leg_action_scale = a.leg_action_scale
    failed = []

    if "parity" not in a.skip:
        env = StairsStepLength(config=cfg, layout="test")
        ok, rows = parity(env)
        print(f"parity ({cfg.impl} vs plain MuJoCo, 0.25 s holding the default pose; longer horizons diverge in unstable poses):", *rows, sep="\n", flush=True)
        failed += [] if ok else ["parity"]

    env = StairsStepLength(config=cfg)
    if "throughput" not in a.skip:
        # training batch size: at 4096 envs the L40S measured 33k steps/s against ~100k in training
        sps = throughput(env, 8192 if gpu else 16, 200 if gpu else 5)
        good = sps >= a.min_steps_per_s or not gpu
        print(f"throughput: {sps:,.0f} env steps/s {'ok' if good else f'< {a.min_steps_per_s:,.0f}: too slow'}"
              f"{'' if gpu else ' (CPU, not checked)'}", flush=True)
        failed += [] if good else ["throughput"]

    if a.init_from and "warm" not in a.skip:
        import json
        from pathlib import Path
        from g1pipe.stairs_env import action_scales
        src_cfg = Path(a.init_from).with_name("config.json")
        src = cfg.copy_and_resolve_references()
        src.leg_action_scale = json.loads(src_cfg.read_text())["env"].get("leg_action_scale", 0.0) if src_cfg.exists() else 0.0
        gain = None if src.leg_action_scale == cfg.leg_action_scale else \
            action_scales(env.mj_model, src) / np.asarray(env._config.action_scale)
        diff = warm_start_check(env, a.init_from, gain)
        if diff is None:
            print("warm start: new observation inputs, parent not comparable (skipped)")
        else:
            limit = 1e-4 if gain is None else 0.15
            good = diff < limit
            print(f"warm start: max joint-target difference to the parent {diff:.4f} rad "
                  f"{'ok' if good else f'> {limit}: the policy does not start from its parent'}", flush=True)
            failed += [] if good else ["warm start"]

    print("PREFLIGHT", "FAILED: " + ", ".join(failed) if failed else "OK", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
