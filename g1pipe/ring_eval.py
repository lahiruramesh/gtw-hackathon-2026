"""Evaluate bearing-ring pick-and-drop policies (g1pipe.ring_env): batched episodes in MJX with 95 %
intervals, and an annotated video.

Per episode (a new tray position, fixture position and ring yaw each time), at the end:
  success    lifted, placed on the fixture (centre within PLACE_TOL, resting, upright) and released
  lifted     the ring was 4+ cm off the table at some point
  placed     on the fixture at the end, released or not
  raceway    a finger pad was inside the bore at some point (raceway-contact proxy)
  dropped    the ring left the table area
The report's target for this skill (docs/report §11) is pick success >= 99.5 % over 1000 attempts
with no raceway contact.

    python -m g1pipe.ring_eval runs/g1-ring-v1/run/params.pkl --episodes 1024 --out runs/g1-ring-v1/run/eval.json
    python -m g1pipe.ring_eval runs/g1-ring-v1/run/params.pkl --video results/videos/ring_v1.mp4 --video-episodes 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from g1pipe import ring_env as R
from g1pipe.gpu_eval import load_params, wilson


def env_module(params_path):
    """The ring task the policy was trained on: the Franka stand-in (ring) or the G1 humanoid (g1ring)."""
    f = Path(params_path).with_name("config.json")
    task = json.loads(f.read_text()).get("task", "ring") if f.exists() else "ring"
    if task == "g1ring":
        from g1pipe import g1_ring_env as G
        return G, G.G1RingPickDrop
    return R, R.RingPickDrop


def make_env(impl=None, n=1, params_path=None):
    mod, cls = env_module(params_path) if params_path else (R, R.RingPickDrop)
    cfg = mod.default_config()
    cfg.impl = impl or ("warp" if jax.default_backend() == "gpu" else "jax")
    per_env = cfg.naconmax // 2048
    cfg.naconmax, cfg.naccdmax = per_env * max(n, 1), per_env * max(n, 1)
    return cls(cfg)


def make_policy_fn(ref):
    nets = ppo_networks.make_ppo_networks(ref["obs_size"], ref["action_size"],
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **ref["network_factory"])
    return ppo_networks.make_inference_fn(nets)


def evaluate(paths, episodes=1024, seed=0, impl=None, log=print):
    env = make_env(impl, episodes, paths[0])
    make_policy = make_policy_fn(load_params(paths[0]))
    n_steps = env._config.episode_length

    def rollout(params, key):
        policy = make_policy(params, deterministic=True)
        state = env.reset(key)

        def body(carry, _):
            state, raceway, dropped = carry
            act, _ = policy(state.obs, jax.random.PRNGKey(0))
            # no per-env select over the state: MuJoCo Warp's contact buffers are shared by the batch
            # (see stairs_env.step); a ring that left the table stays a failure through `dropped`
            nxt = env.step(state, act)
            return (nxt, raceway | (nxt.metrics["raceway"] > 0), dropped | (nxt.done > 0)), None

        init = (state, jp.zeros((), bool), jp.zeros((), bool))
        (state, raceway, dropped), _ = jax.lax.scan(body, init, None, length=n_steps)
        out = env.outcome(state.data, state.info)
        return {"success": out["released"] & out["lifted"] & ~dropped, "lifted": out["lifted"],
                "placed": out["placed"] & ~dropped, "raceway": raceway, "dropped": dropped, "xy_err": out["xy_err"]}

    batched = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))
    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    results = []
    for p in paths:
        r = {k: np.asarray(v) for k, v in batched(load_params(p)["params"], keys).items()}
        row = {"params": str(p), "episodes": episodes}
        for k in ("success", "lifted", "placed", "raceway", "dropped"):
            n = int(r[k].sum())
            row[k] = n
            row[f"{k}_rate"] = n / episodes
            row[f"{k}_ci95"] = list(wilson(n, episodes))
        row["xy_err_median_mm"] = float(np.median(r["xy_err"]) * 1000)
        lo, hi = row["success_ci95"]
        log(f"{Path(p).name:24} success {row['success']}/{episodes} ({row['success_rate']*100:.1f} %, CI {lo*100:.1f}-"
            f"{hi*100:.1f}) lifted {row['lifted']} placed {row['placed']} raceway {row['raceway']} dropped {row['dropped']}")
        results.append(row)
    results.sort(key=lambda x: (-x["success"], x["raceway"]))
    return results


# -- video ----------------------------------------------------------------------------------------

def _font(size, bold=False):
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size, index=1 if bold and p.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def make_video(path, out, episodes=6, seed=1, label=None, fps=25):
    import imageio.v2 as imageio
    import mujoco
    from mujoco import mjx
    from PIL import Image, ImageDraw

    W, H, VIEW_W = 1280, 720, 880
    BG, FG, DIM, ACCENT, OK, BAD = (18, 20, 24), (236, 238, 241), (150, 156, 165), (86, 156, 255), (80, 200, 120), (235, 90, 80)
    fh, fb, fs, fbig = _font(19, True), _font(17), _font(14), _font(24, True)
    env = make_env("jax", 1, path)
    humanoid = type(env).__name__ == "G1RingPickDrop"
    bench_z = env_module(path)[0].BENCH_Z if humanoid else 0.0
    ref = load_params(path)
    policy = jax.jit(make_policy_fn(ref)(ref["params"], deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    m = env.mj_model
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(m, H, VIEW_W)
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    if humanoid:
        cam.lookat[:] = (0.25, -0.06, 0.88)
        cam.distance, cam.azimuth, cam.elevation = 1.55, 145, -22
    else:
        cam.lookat[:] = (0.42, 0.0, 0.22)
        cam.distance, cam.azimuth, cam.elevation = 1.75, 205, -24
    label = label or Path(path).parent.parent.name
    writer = imageio.get_writer(out, fps=fps, quality=8, macro_block_size=8)
    every = max(1, int(round(1 / (fps * env._config.ctrl_dt))))
    results = []

    def panel(img, ep, t, stage, info, status):
        dr = ImageDraw.Draw(img)
        dr.rectangle([VIEW_W, 0, W, H], fill=BG)
        x, y = VIEW_W + 24, 20
        dr.text((x, y), f"Bearing ring pick and drop · {label}", font=fh, fill=FG); y += 26
        dr.text((x, y), "Unitree G1 humanoid, right arm + Dex3 hand" if humanoid else
                "Franka arm + parallel gripper (stand-in for G1 arm/hand)", font=fs, fill=DIM); y += 34
        rows = [("PART", None), ("Outer ring, 6206 class", ""), ("Outside diameter", f"{R.RING_OD*1000:.0f} mm"),
                ("Bore (raceway side)", f"{R.RING_ID*1000:.0f} mm"), ("Width", f"{R.RING_W*1000:.0f} mm"),
                ("Mass", f"{R.RING_MASS:.2f} kg"), ("TASK", None),
                ("Pick from", "tray (blue)"), ("Drop on", "fixture (orange)"),
                ("Tray to fixture", f"{info['dist']*100:.0f} cm"), ("Place tolerance", f"{R.PLACE_TOL*1000:.0f} mm"),
                ("LIVE", None), ("Episode", f"{ep + 1} / {episodes}"), ("Time", f"{t:.1f} s"),
                ("Ring above the table", f"{info['z']*1000:.0f} mm"), ("Ring to fixture", f"{info['xy']*1000:.0f} mm"),
                ("Finger in bore", "yes" if info["bore"] else "no")]
        for k, v in rows:
            if v is None:
                y += 6
                dr.text((x, y), k, font=fs, fill=ACCENT); y += 22
                continue
            dr.text((x, y), k, font=fs, fill=DIM)
            dr.text((W - 24, y), v, font=fb, fill=FG, anchor="ra"); y += 22
        y += 10
        dr.text((x, y), f"Stage: {stage}", font=fb, fill=FG); y += 30
        col = {"success": OK, "failed": BAD, "running": FG}[status]
        dr.text((x, y), {"success": "PLACED AND RELEASED", "failed": "NOT PLACED", "running": ""}[status], font=fbig, fill=col)
        dr.text((x, H - 30), "MuJoCo Playground (MJX), deterministic policy", font=fs, fill=DIM)
        return img

    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    for ep in range(episodes):
        s = reset(keys[ep])
        target = np.asarray(s.info["target_pos"])
        body = env._ring if humanoid else env._obj_body
        start = np.asarray(s.data.xpos[body])
        dist = float(np.linalg.norm(target[:2] - start[:2]))
        bore = False
        last = None
        for k in range(env._config.episode_length):
            act = policy(s.obs, jax.random.PRNGKey(0))[0]
            s = step(s, act)
            ring = np.asarray(s.data.xpos[body])
            bore = bore or bool(s.metrics["raceway"] > 0)
            out = {kk: bool(v) if v.dtype == bool else float(v) for kk, v in env.outcome(s.data, s.info).items()}
            lifted = float(s.info["lifted"]) > 0
            stage = ("released" if out["released"] else "placing" if out["placed"] else "carrying" if lifted
                     else "grasping" if float(s.info["reached"]) > 0 else "reaching")
            if k % every == 0 or k == env._config.episode_length - 1:
                dd = mjx.get_data(m, s.data)
                d.qpos[:], d.qvel[:], d.mocap_pos[:], d.mocap_quat[:] = dd.qpos, dd.qvel, dd.mocap_pos, dd.mocap_quat
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera=cam)
                img = Image.new("RGB", (W, H), BG)
                img.paste(Image.fromarray(renderer.render()), (0, 0))
                info = {"dist": dist, "z": ring[2] - bench_z - R.RING_W / 2, "xy": out["xy_err"], "bore": bore}
                last = (img, (k + 1) * env._config.ctrl_dt, stage, info)
                writer.append_data(np.asarray(panel(img.copy(), ep, *last[1:], "running")))
            if float(s.done) > 0:
                break
        success = out["released"] and lifted and float(s.done) == 0
        final = np.asarray(panel(last[0].copy(), ep, *last[1:], "success" if success else "failed"))
        for _ in range(fps):
            writer.append_data(final)
        results.append({"episode": ep, "success": bool(success), "lifted": lifted, "placed": out["placed"],
                        "raceway": bore, "xy_err_mm": round(out["xy_err"] * 1000, 1), "transfer_cm": round(dist * 100, 1)})
        print(json.dumps(results[-1]), flush=True)
    writer.close()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", nargs="+")
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--out", default=None, help="JSON with every result, best first")
    ap.add_argument("--video", default=None, help="render the first params file to this mp4 (CPU)")
    ap.add_argument("--video-episodes", type=int, default=6)
    a = ap.parse_args()
    if a.video:
        res = make_video(a.params[0], a.video, a.video_episodes)
        Path(a.video).with_suffix(".json").write_text(json.dumps(res, indent=1))
        print("video:", a.video)
        return
    results = evaluate(a.params, a.episodes, a.seed, a.impl, log=lambda m: print(m, flush=True))
    print("\nbest:", results[0]["params"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
