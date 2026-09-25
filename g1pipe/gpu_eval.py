"""Batched held-out stairs evaluation in MJX: thousands of crossings in a minute on the training GPU.

Plain-MuJoCo evaluation (g1pipe.stairs_eval) runs one crossing at a time on the Mac, so a
32-crossing test was all we could afford per checkpoint, and it swings by +-3 crossings between
near-identical policies. Since v9 the training and test physics agree on foot contacts, so the
training engine can screen checkpoints with a sample large enough to rank them; the plain-MuJoCo
suite (scripts/certify.py) stays the final gate for the finalists.

Every env starts on a held-out ("test" layout) staircase of a uniformly random level, square to
the steps, with a fixed command (vx, 1.4 Hz stairs cadence, hold the heading) and the scan model
the policy will be deployed with. Per episode: crossed (walked past the last step), fell (fall or
turned away, at any time in the episode). Rates come with 95 % Wilson intervals.

    python -m g1pipe.gpu_eval runs/g1-stairs-v13/run/ckpt_*.pkl --episodes 2048 --scan camera
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from g1pipe import stairs_terrain as T


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a rate k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_params(path):
    """params.pkl, or a ckpt_*.pkl next to one (raw params; network shape from the run's params.pkl)."""
    blob = pickle.load(open(path, "rb"))
    if not (isinstance(blob, dict) and "params" in blob):
        blob = {**pickle.load(open(Path(path).with_name("params.pkl"), "rb")), "params": blob}
    return blob


def run_config(path):
    """Env settings the policy was trained with (config.json next to the params)."""
    f = Path(path).with_name("config.json")
    return json.loads(f.read_text())["env"] if f.exists() else {}


def make_env(path, scan="true", vx=0.7, impl=None):
    from g1pipe.stairs_env import STAIRS_CADENCE, StairsStepLength, default_config
    cfg = default_config()
    cfg.impl = impl or ("warp" if jax.default_backend() == "gpu" else "jax")
    trained = run_config(path)
    cfg.leg_action_scale = trained.get("leg_action_scale", 0.0)
    cfg.action_delay_max = trained.get("action_delay_max", 0)   # rank as trained (no latency before v14)
    cfg.action_delay_p = trained.get("action_delay_p", 0.5)     # v14 drew 0 or 1 step uniformly
    cfg.scan_model = "camera" if scan == "camera" else "uniform"
    cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_yaw = [vx, vx], [0.0, 0.0], [0.0, 0.0]
    cfg.gait_freq_range = [STAIRS_CADENCE, STAIRS_CADENCE]
    return StairsStepLength(config=cfg, layout="test", eval_levels=True), STAIRS_CADENCE


def make_evaluator(ref_path, episodes=2048, seconds=10.0, scan="true", vx=0.7, seed=0, impl=None):
    """Compile one batched rollout for the network shape of `ref_path`; returns f(path) -> result.
    Checkpoints of the same run share the network, so ranking a whole run costs one compile."""
    env, f = make_env(ref_path, scan, vx, impl)
    ref = load_params(ref_path)
    nets = ppo_networks.make_ppo_networks(ref["obs_size"], ref["action_size"],
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **ref["network_factory"])
    make_policy = ppo_networks.make_inference_fn(nets)
    cmd = jp.array([vx, 0.0, 0.0])

    def pin(state):
        # Playground zeroes 10 % of commands and resamples every 500 steps; hold vx and the cadence
        info = dict(state.info, command=state.info["command"].at[:2].set(cmd[:2]),
                    gait_freq=jp.asarray(f), step_cmd=jp.asarray(vx / (2 * f)),
                    phase_dt=2 * jp.pi * env.dt * jp.array([f]))
        return state.replace(info=info)

    def rollout(params, key):
        policy = make_policy(params, deterministic=True)
        state = pin(env.reset(key))

        def body(carry, _):
            state, fell, crossed = carry
            act, _ = policy(state.obs, jax.random.PRNGKey(0))
            state = pin(env.step(state, act))
            cur = state.info["curriculum"]
            fell = fell | cur["fell"]
            crossed = crossed | (cur["crossed"] & ~fell)   # crossed before any fall
            return (state, fell, crossed), None

        init = (state, jp.zeros((), bool), jp.zeros((), bool))
        (state, fell, crossed), _ = jax.lax.scan(body, init, None, length=int(seconds / env.dt))
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
        return {"params": str(path), "scan": scan, "vx": vx, "episodes": n, "seconds": seconds,
                "crossed": int(crossed.sum()), "fell": k, "fall_rate": k / n, "fall_ci95": list(wilson(k, n)),
                "per_level": rows}

    return run


def evaluate(path, episodes=2048, seconds=10.0, scan="true", vx=0.7, seed=0, impl=None):
    return make_evaluator(path, episodes, seconds, scan, vx, seed, impl)(path)


def summary(r) -> str:
    per = " ".join(f"{x['rise_cm']}:{x['crossed']}/{x['n']}f{x['fell']}" for x in r["per_level"])
    lo, hi = r["fall_ci95"]
    return (f"{Path(r['params']).parent.parent.name}/{Path(r['params']).name:>22} {r['scan']:6} | "
            f"crossed {r['crossed']}/{r['episodes']} fell {r['fell']} ({r['fall_rate']*100:.1f} %, 95 % CI "
            f"{lo*100:.1f}-{hi*100:.1f}) | {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="+", help="params.pkl / ckpt_*.pkl files; all are ranked")
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--scan", default="true", choices=["true", "camera"])
    ap.add_argument("--vx", type=float, default=0.7)
    ap.add_argument("--impl", default=None, help="'warp' (NVIDIA GPU) or 'jax'")
    ap.add_argument("--out", default=None, help="JSON with every result, best first")
    a = ap.parse_args()
    run = make_evaluator(a.params[0], a.episodes, a.seconds, a.scan, a.vx, impl=a.impl)
    results = []
    for p in a.params:
        r = run(p)
        print(summary(r), flush=True)
        results.append(r)
    results.sort(key=lambda r: (r["fell"], -r["crossed"]))
    print("\nbest:", results[0]["params"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
