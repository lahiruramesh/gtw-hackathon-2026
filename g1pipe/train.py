"""Train the G1 step-length policy with Brax PPO (MuJoCo Playground recipe).

Runs anywhere JAX runs: Kaggle/Colab GPU for real training, the Mac CPU for a
smoke test.

    python -m g1pipe.train --timesteps 150_000_000 --out runs/steplength_v1      # GPU
    python -m g1pipe.train --smoke --out runs/smoke                               # CPU check

Writes to --out:  params.pkl (final), ckpt_*.pkl (periodic), progress.csv, config.json
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import pickle
import time
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize
from mujoco_playground.config import locomotion_params

from g1pipe.steplength_env import StepLength, default_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/steplength")
    ap.add_argument("--timesteps", type=int, default=150_000_000)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--impl", default=None, help="'warp' (NVIDIA GPU) or 'jax'; default picks by backend")
    ap.add_argument("--no-dr", action="store_true", help="disable domain randomisation (experiment E3)")
    ap.add_argument("--step-scale", type=float, default=None, help="override step_length reward scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline end to end")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    print("JAX backend:", backend, "devices:", jax.devices())

    env_cfg = default_config()
    env_cfg.impl = a.impl or ("warp" if backend == "gpu" else "jax")
    if a.step_scale is not None:
        env_cfg.reward_config.scales.step_length = a.step_scale
    if a.no_dr:
        env_cfg.push_config.enable = False
        env_cfg.noise_config.level = 0.0

    rl = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    rl.num_timesteps = a.timesteps
    if a.num_envs:
        rl.num_envs = a.num_envs
    if a.smoke:
        rl.num_timesteps, rl.num_envs, rl.batch_size = 20_000, 32, 32
        rl.num_minibatches, rl.num_evals, rl.episode_length = 4, 2, 100
        rl.num_resets_per_eval = 0
        env_cfg.naconmax, env_cfg.njmax = 64, env_cfg.njmax

    env = StepLength(config=env_cfg)
    eval_env = StepLength(config=env_cfg)

    params_nf = dict(rl.network_factory)
    train_kwargs = {k: v for k, v in rl.items() if k != "network_factory"}
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **params_nf)

    (out / "config.json").write_text(json.dumps({
        "env": env_cfg.to_dict(), "ppo": {**train_kwargs, "network_factory": params_nf},
        "no_dr": a.no_dr, "seed": a.seed, "backend": backend,
    }, indent=2, default=str))

    t0 = time.time()
    log = open(out / "progress.csv", "w", newline="")
    writer = None

    def progress(step, metrics):
        nonlocal writer
        row = {"step": step, "wall_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in metrics.items() if k.startswith("eval/")}}
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row.keys()), extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        log.flush()
        print(f"[{row['wall_s']:>7.0f}s] step {step:>11,}  reward {row.get('eval/episode_reward', float('nan')):8.2f}"
              f"  step_len_err {row.get('eval/episode_step_len_err', float('nan')):.3f}", flush=True)

    def save_ckpt(step, make_policy, params):
        with open(out / f"ckpt_{step:011d}.pkl", "wb") as f:
            pickle.dump(jax.device_get(params), f)

    train_fn = functools.partial(
        ppo.train, **train_kwargs, network_factory=network_factory, seed=a.seed,
        randomization_fn=None if a.no_dr else g1_randomize.domain_randomize,
        progress_fn=progress, policy_params_fn=save_ckpt,
    )
    make_inference_fn, params, _ = train_fn(
        environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)

    with open(out / "params.pkl", "wb") as f:
        pickle.dump({"params": jax.device_get(params), "network_factory": params_nf,
                     "obs_size": env.observation_size, "action_size": env.action_size}, f)
    print(f"done in {time.time() - t0:.0f}s -> {out}/params.pkl")


if __name__ == "__main__":
    main()
