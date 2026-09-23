"""Watch a trained G1 policy walk in a live MuJoCo window on the Mac (CPU, no GPU needed).

    uv run mjpython scripts/view_policy.py                              # v1 (with randomisation)
    uv run mjpython scripts/view_policy.py --run nodr                   # E3 policy (no randomisation)
    uv run mjpython scripts/view_policy.py --params path/to/params.pkl

Keys (click the window first):
    Up / Down      speed  +/- 0.1 m/s
    Right / Left   step length  +/- 2.5 cm
    P              push the robot sideways
    R              reset
On macOS the viewer needs `mjpython` (ships with the mujoco package), not plain `python`.
"""
import argparse
import time

import jax
import mujoco
import mujoco.viewer
import numpy as np

from g1pipe.evaluate import HEADING_KP, PolicyRunner, command_to_gait

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="v1", help="v1 | nodr | any runs/g1-steplength-<name>")
ap.add_argument("--params", default=None)
ap.add_argument("--vx", type=float, default=0.6)
ap.add_argument("--step", type=float, default=0.20)
ap.add_argument("--duration", type=float, default=0, help="auto-close after N seconds (0 = until closed)")
a = ap.parse_args()
params = a.params or f"runs/g1-steplength-{a.run}/run/params.pkl"

r = PolicyRunner(params)
m, d = r.m, r.d
cmd = {"vx": a.vx, "step": a.step, "push": False, "reset": True}


def on_key(key):
    if key == 265: cmd["vx"] = min(1.0, cmd["vx"] + 0.1)          # up
    elif key == 264: cmd["vx"] = max(0.0, cmd["vx"] - 0.1)        # down
    elif key == 262: cmd["step"] = min(0.40, cmd["step"] + 0.025)  # right
    elif key == 263: cmd["step"] = max(0.10, cmd["step"] - 0.025)  # left
    elif key == ord("P"): cmd["push"] = True
    elif key == ord("R"): cmd["reset"] = True


def reset():
    mujoco.mj_resetData(m, d)
    d.qpos[:] = r.init_q
    d.ctrl[:] = r.init_q[7:]
    mujoco.mj_forward(m, d)
    z = np.zeros(m.nu, np.float32)
    return {"phase": np.array([0.0, np.pi]), "obs_phase": np.array([0.0, np.pi]),
            "obs_act": z, "prev_act": z, "yaw0": None, "air": np.zeros(2), "prev_c": np.array([True, True])}


print(__doc__.split("Keys")[1])
with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
    viewer.cam.distance, viewer.cam.elevation = 3.0, -12
    st, t0, last_print = None, time.time(), 0.0
    while viewer.is_running() and (a.duration == 0 or time.time() - t0 < a.duration):
        tick = time.time()
        if cmd["reset"]:
            st, cmd["reset"] = reset(), False
        f, step_cmd = command_to_gait(cmd["vx"], cmd["step"])
        R = d.xmat[r.torso].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        st["yaw0"] = yaw if st["yaw0"] is None else st["yaw0"]
        wz = float(np.clip(HEADING_KP * np.angle(np.exp(1j * (st["yaw0"] - yaw))), -0.5, 0.5))
        obs = r._obs(np.array([cmd["vx"], 0.0, wz]), st["obs_act"], st["obs_phase"], step_cmd, f)
        act, _ = r._policy(obs, jax.random.PRNGKey(0))
        act = np.asarray(act, np.float32)
        st["obs_act"], st["prev_act"] = st["prev_act"], act
        d.ctrl[:] = r.default + act * r.action_scale
        for _ in range(r.n_sub):
            mujoco.mj_step(m, d)
        if cmd["push"]:
            d.qvel[1] += 0.6
            cmd["push"] = False
        st["obs_phase"] = st["phase"]
        st["phase"] = np.fmod(st["phase"] + 2 * np.pi * r.ctrl_dt * f + np.pi, 2 * np.pi) - np.pi

        # live step-length readout
        c = np.array([d.sensordata[x] > 0 for x in r.floor_sensors])
        td = c & ~st["prev_c"] & (st["air"] >= 0.1)
        if td.any():
            fwd = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
            feet = d.site_xpos[r.feet_sites][:, :2]
            i = int(np.argmax(td))
            ach = float(np.dot(feet[i] - feet[1 - i], fwd))
            if d.time - last_print > 0.3:
                print(f"speed cmd {cmd['vx']:.1f} m/s  step cmd {step_cmd*100:4.1f} cm  achieved {ach*100:5.1f} cm", flush=True)
                last_print = d.time
        st["air"] = np.where(c, 0.0, st["air"] + r.ctrl_dt)
        st["prev_c"] = c
        if R[2, 2] < 0.3:
            print("fell — resetting")
            cmd["reset"] = True

        viewer.cam.lookat[:] = d.qpos[:3]
        viewer.sync()
        time.sleep(max(0.0, r.ctrl_dt - (time.time() - tick)))
