"""Step-length tracking measured in the *training* engine (MJX, jax impl on CPU).

Separates "did the policy learn the task?" from "does it transfer to another engine?"
(g1pipe/evaluate.py measures the latter in MuJoCo C).

    uv run python scripts/eval_mjx.py runs/g1-steplength-v1/run/params.pkl --tag v1
"""
import argparse, csv, json, pickle
from pathlib import Path
import jax, jax.numpy as jp, numpy as np
from mujoco import mjx
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from g1pipe.steplength_env import GAIT_FREQ_RANGE, StepLength, default_config

ap = argparse.ArgumentParser()
ap.add_argument("params"); ap.add_argument("--tag", default="latest")
ap.add_argument("--no-dr-env", action="store_true")
a = ap.parse_args()
blob = pickle.load(open(a.params, "rb"))
cfg = default_config(); cfg.impl = "jax"; cfg.noise_config.level = 0.0; cfg.push_config.enable = False
env = StepLength(config=cfg)
nets = ppo_networks.make_ppo_networks(blob["obs_size"], blob["action_size"],
                                      preprocess_observations_fn=running_statistics.normalize, **blob["network_factory"])
pol = jax.jit(ppo_networks.make_inference_fn(nets)(blob["params"], deterministic=True))
reset, step = jax.jit(env.reset), jax.jit(env.step)
feet, torso = np.array(env._feet_site_id), env._torso_body_id


def rollout(vx, sl, n=450, settle=100):
    f = float(np.clip(vx / (2 * sl), *GAIT_FREQ_RANGE)); cmd_step = vx / (2 * f)
    def fix(s):
        i = dict(s.info); i.update(command=jp.array([vx, 0., 0.]), gait_freq=jp.array(f),
                                   step_cmd=jp.array(cmd_step), phase_dt=2 * jp.pi * env.dt * jp.array([f]))
        return s.replace(info=i)
    s = reset(jax.random.PRNGKey(0))
    q0 = np.array(env._init_q)
    d = mjx.forward(env.mjx_model, s.data.replace(qpos=jp.array(q0), qvel=jp.zeros_like(s.data.qvel), ctrl=jp.array(q0[7:])))
    i = dict(s.info); i.update(phase=jp.array([0., jp.pi]), last_act=jp.zeros(env.action_size))
    s = fix(s.replace(data=d, info=i)); s = s.replace(obs=env._get_obs(s.data, s.info, env._contact(s.data)))
    prev, air, steps, vxs = np.array([True, True]), np.zeros(2), [], []
    for k in range(n):
        act, _ = pol(s.obs, jax.random.PRNGKey(0)); s = fix(step(s, act))
        if float(s.done): return {"fell": True}
        c = np.array(env._contact(s.data)); td = c & ~prev & (air >= 0.1)
        if k >= settle:
            R = np.array(s.data.xmat[torso]); fwd = R[:2, 0] / np.linalg.norm(R[:2, 0])
            p = np.array(s.data.site_xpos[feet])[:, :2]
            for j in (0, 1):
                if td[j]: steps.append(float(np.dot(p[j] - p[1 - j], fwd)))
            vxs.append(float(env.get_local_linvel(s.data, "pelvis")[0]))
        air = np.where(c, 0., air + env.dt); prev = c
    st = np.array(steps)
    return {"fell": False, "step_cmd": cmd_step, "step_ach": float(st.mean()), "step_err": float(np.abs(st - cmd_step).mean()),
            "step_sd": float(st.std()), "vx_err": float(np.abs(np.array(vxs) - vx).mean())}


rows = []
for vx in (0.5, 0.7, 0.9):
    for sl in (0.15, 0.25, 0.35):
        r = {"vx": vx, "step_req": sl, **rollout(vx, sl)}; rows.append(r)
        print("MJX", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
out = Path("results") / f"eval_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
keys = sorted({k for r in rows for k in r})
with open(out / "mjx_grid.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if not r["fell"]]
print(json.dumps({"mjx_fall_rate": 1 - len(ok) / len(rows), "mjx_step_err_cm": 100 * float(np.mean([r["step_err"] for r in ok])),
                  "mjx_vx_err": float(np.mean([r["vx_err"] for r in ok]))}))
