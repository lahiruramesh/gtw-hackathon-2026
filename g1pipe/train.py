"""Train the G1 step-length policy with Brax PPO (MuJoCo Playground recipe).

Runs anywhere JAX runs: Kaggle/Colab GPU for real training, the Mac CPU for a
smoke test.

    python -m g1pipe.train --timesteps 150_000_000 --out runs/steplength_v1      # GPU
    python -m g1pipe.train --smoke --out runs/smoke                               # CPU check
    python -m g1pipe.train --task stairs --timesteps 200_000_000 --out runs/stairs_v1   # stairs + height scan
    python -m g1pipe.train --task stairs --init-from runs/g1-steplength-v1/run/params.pkl --out runs/stairs_v2
                                                   # stairs, warm-started from the flat policy
    python -m g1pipe.train --task ring --timesteps 60_000_000 --out runs/ring_v1
                                                   # bearing ring pick and drop (g1pipe.ring_env, Franka stand-in)

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
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize
from mujoco_playground.config import locomotion_params, manipulation_params

from g1pipe.steplength_env import StepLength, default_config


def warm_start(path, obs_size, action_gain=None):
    """Brax restore_params from a trained policy whose observations are a prefix of ours.

    The flat step-length policy's observations are the stairs policy's minus the trailing
    height scan. New inputs get zero weights in the first layer (so the warm-started policy
    starts out walking exactly like the flat one) and normaliser stats from the terrain.
    """
    from g1pipe import stairs_terrain as T
    blob = pickle.load(open(path, "rb"))
    if not (isinstance(blob, dict) and "params" in blob):
        # a periodic checkpoint (ckpt_*.pkl) holds raw params; obs sizes come from the run's params.pkl
        blob = {**pickle.load(open(Path(path).with_name("params.pkl"), "rb")), "params": blob}
    norm, policy, value = blob["params"]
    mean, std = T.scan_stats()
    count = float(norm.count.to_numpy())
    new = {"mean": dict(norm.mean), "std": dict(norm.std), "summed_variance": dict(norm.summed_variance)}
    for key, (size,) in obs_size.items():
        extra = size - blob["obs_size"][key][0]
        if extra == 0:
            continue
        if extra != T.N_SCAN:
            raise ValueError(f"{key}: {extra} new inputs, expected the {T.N_SCAN}-point height scan")
        new["mean"][key] = np.concatenate([norm.mean[key], mean])
        new["std"][key] = np.concatenate([norm.std[key], std])
        new["summed_variance"][key] = np.concatenate([norm.summed_variance[key], std ** 2 * count])
    norm = norm.replace(**new)

    def pad_first_layer(p, extra):
        k = p["params"]["hidden_0"]["kernel"]
        p = jax.tree.map(lambda x: x, p)
        p["params"]["hidden_0"]["kernel"] = np.concatenate([k, np.zeros((extra, k.shape[1]), k.dtype)])
        return p

    policy = pad_first_layer(policy, obs_size["state"][0] - blob["obs_size"]["state"][0])
    if action_gain is not None:
        # wider action scale on some joints: shrink the policy's mean output there by the same
        # factor, so the joint targets (default + scale * tanh(mean)) start out nearly unchanged
        out = max(policy["params"], key=lambda k: int(k.split("_")[-1]))
        layer = dict(policy["params"][out])
        gain = np.concatenate([action_gain, np.ones_like(action_gain)])   # [mean | std] outputs
        layer["kernel"] = np.asarray(layer["kernel"]) * gain
        layer["bias"] = np.asarray(layer["bias"]) * gain
        policy["params"][out] = layer
    value = pad_first_layer(value, obs_size["privileged_state"][0] - blob["obs_size"]["privileged_state"][0])
    return norm, policy, value


def parse_overrides(items):
    """["env.a.b=0.7", "ppo.learning_rate=2e-4"] -> {"env.a.b": 0.7, ...}; values are JSON, else strings."""
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.startswith(("env.", "ppo.")):
            raise SystemExit(f"--set {item!r}: expected env.<path>=VALUE or ppo.<name>=VALUE")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def apply_overrides(env_cfg, rl, overrides):
    """Set each override on the env config or the PPO config; unknown keys are an error, so a typo
    in an experiment never trains the unchanged recipe."""
    for key, value in overrides.items():
        root, *path = key.split(".")
        if root not in ("env", "ppo") or not path:
            raise SystemExit(f"override {key!r}: expected env.<path> or ppo.<name>")
        node = env_cfg if root == "env" else rl
        for part in path[:-1]:
            if part not in node:
                raise SystemExit(f"--set {key}: no config field {part!r}")
            node = node[part]
        if path[-1] not in node:
            raise SystemExit(f"--set {key}: no config field {path[-1]!r}")
        old = node[path[-1]]
        if isinstance(old, float) and isinstance(value, int):
            value = float(value)
        node[path[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/steplength")
    ap.add_argument("--task", choices=["flat", "stairs", "ring", "g1ring"], default="flat",
                    help="stairs: g1pipe.stairs_env (stair terrain + height scan observation)")
    ap.add_argument("--timesteps", type=int, default=150_000_000)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--impl", default=None, help="'warp' (NVIDIA GPU) or 'jax'; default picks by backend")
    ap.add_argument("--no-dr", action="store_true", help="disable domain randomisation (experiment E3)")
    ap.add_argument("--step-scale", type=float, default=None, help="override step_length reward scale")
    ap.add_argument("--scan-model", default=None, choices=["uniform", "camera"],
                    help="stairs: height-scan noise; camera = head depth camera + elevation map errors")
    ap.add_argument("--leg-action-scale", type=float, default=None,
                    help="stairs: action scale of hip pitch, knee, ankle pitch (Playground: 0.5)")
    ap.add_argument("--lr", type=float, default=None, help="PPO learning rate (Playground G1: 3e-4); lower to fine-tune")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="params.pkl of a trained policy to start from (stairs: the flat step-length policy)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value: env.<dotted path> or ppo.<name>, VALUE as JSON "
                         "(e.g. env.action_delay_p=0.7, env.payload_kg=[0,4], ppo.learning_rate=2e-4); repeatable")
    ap.add_argument("--overrides", default=None,
                    help='the same as --set, as one JSON object: {"env.action_delay_p": 0.7} (the web app\'s form field)')
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline end to end")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    print("JAX backend:", backend, "devices:", jax.devices())

    if a.task == "stairs":
        from g1pipe.stairs_env import StairsStepLength as Env, default_config as env_default_config
        from g1pipe.stairs_env import domain_randomize as stairs_randomize
    elif a.task == "ring":
        from g1pipe.ring_env import RingPickDrop as Env, default_config as env_default_config
        stairs_randomize = None
    elif a.task == "g1ring":
        from g1pipe.g1_ring_env import G1RingPickDrop as Env, default_config as env_default_config
        stairs_randomize = None
    else:
        Env, env_default_config = StepLength, default_config
        stairs_randomize = None
    env_cfg = env_default_config()
    env_cfg.impl = a.impl or ("warp" if backend == "gpu" else "jax")
    if a.step_scale is not None:
        env_cfg.reward_config.scales.step_length = a.step_scale
    if a.scan_model:
        env_cfg.scan_model = a.scan_model
    if a.leg_action_scale:
        env_cfg.leg_action_scale = a.leg_action_scale
    if a.no_dr and a.task not in ("ring", "g1ring"):
        env_cfg.push_config.enable = False
        env_cfg.noise_config.level = 0.0

    if a.task in ("ring", "g1ring"):   # Playground's tuned recipe for its Panda pick task, our episode length
        rl = manipulation_params.brax_ppo_config("PandaPickCube")
        rl.episode_length = env_cfg.episode_length
    else:
        rl = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    rl.num_timesteps = a.timesteps
    if a.num_envs:
        rl.num_envs = a.num_envs
    if a.lr:
        rl.learning_rate = a.lr
    if a.smoke:
        rl.num_timesteps, rl.num_envs, rl.batch_size = 20_000, 32, 32
        rl.num_minibatches, rl.num_evals, rl.episode_length = 4, 2, 100
        rl.num_resets_per_eval = 0
        env_cfg.naconmax, env_cfg.njmax = 64, env_cfg.njmax

    overrides = {**(json.loads(a.overrides) if a.overrides else {}), **parse_overrides(a.set)}
    apply_overrides(env_cfg, rl, overrides)

    env = Env(config=env_cfg)
    if a.task == "stairs":
        eval_env = Env(config=env_cfg, eval_levels=True)   # eval on every level, not the curriculum's
        rl.num_resets_per_eval = 0                         # host resets would wipe curriculum levels
    else:
        eval_env = Env(config=env_cfg)
    gain = None
    if a.init_from and a.task == "stairs":
        # rescale the warm-start policy's outputs if it was trained with other per-joint action scales
        from g1pipe.stairs_env import action_scales
        src_cfg = Path(a.init_from).with_name("config.json")
        src = env_cfg.copy_and_resolve_references()
        src.leg_action_scale = json.loads(src_cfg.read_text())["env"].get("leg_action_scale", 0.0) if src_cfg.exists() else 0.0
        if src.leg_action_scale != env_cfg.leg_action_scale:
            gain = action_scales(env.mj_model, src) / np.asarray(env._config.action_scale)
    if a.task in ("ring", "g1ring"):   # same observations and actions: restore the parameters as they are
        restore = pickle.load(open(a.init_from, "rb"))["params"] if a.init_from else None
    else:
        restore = warm_start(a.init_from, env.observation_size, gain) if a.init_from else None

    params_nf = dict(rl.network_factory)
    train_kwargs = {k: v for k, v in rl.items() if k != "network_factory"}
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **params_nf)

    (out / "config.json").write_text(json.dumps({
        "env": env_cfg.to_dict(), "ppo": {**train_kwargs, "network_factory": params_nf},
        "task": a.task, "no_dr": a.no_dr, "seed": a.seed, "backend": backend, "init_from": a.init_from,
        "overrides": overrides,
    }, indent=2, default=str))
    if stairs_randomize is not None:
        stairs_randomize = functools.partial(stairs_randomize, friction_range=tuple(env_cfg.friction_range),
                                             payload_kg=tuple(env_cfg.payload_kg))

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
              f"  step_len_err {row.get('eval/episode_step_len_err', float('nan')):.3f}"
              + (f"  crossed {row['eval/episode_crossed']:.2f}" if "eval/episode_crossed" in row else ""), flush=True)

    def save_ckpt(step, make_policy, params):
        with open(out / f"ckpt_{step:011d}.pkl", "wb") as f:
            pickle.dump(jax.device_get(params), f)

    train_fn = functools.partial(
        ppo.train, **train_kwargs, network_factory=network_factory, seed=a.seed,
        randomization_fn=None if a.no_dr or a.task in ("ring", "g1ring") else (
            stairs_randomize if a.task == "stairs" else g1_randomize.domain_randomize),
        progress_fn=progress, policy_params_fn=save_ckpt, restore_params=restore,
    )
    make_inference_fn, params, _ = train_fn(
        environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)

    with open(out / "params.pkl", "wb") as f:
        pickle.dump({"params": jax.device_get(params), "network_factory": params_nf,
                     "obs_size": env.observation_size, "action_size": env.action_size, "task": a.task}, f)
    print(f"done in {time.time() - t0:.0f}s -> {out}/params.pkl")


if __name__ == "__main__":
    main()
