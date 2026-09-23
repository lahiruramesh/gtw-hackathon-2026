# Jev agent: a self-learning supervisor for the G1 step-length policy

A high-level agent that watches the robot's sensors, asks TypeSafe's **Jev**
(a "System One" decision model: typed questions in, calibrated probabilities out)
what gait to use, decides in code, and learns from its own outcomes.

```
 MuJoCo sensors (50 Hz)                 perception.py   numbers -> named buckets ("feet slipping")
   IMU, gyro, joint/contacts, feet ───►  + operator mission text
                                              │  2 Hz
                                              ▼
                       Jev (jev-1.13.0) ─ one request, six questions (questions.py)
                         gait_mode (Choice) · stability (Score) · slipping / disturbed /
                         mission_speed / mission_care (Noul)
                                              │  probabilities + confidence
                                              ▼
            supervisor.py  1. safety filter in code (tilt > 25°, stability < 0.75 -> stop)
                           2. log P_jev(mode) + learned bonus, minus risk-model vetoes
                           3. confidence gate: slow down any time, speed up one level at conf >= 0.6
                                              │  (speed, step length)
                                              ▼
                PPO step-length policy (g1pipe, 50 Hz) ──► joint targets
                                              │  outcome: progress, tracking, tilt, fall
                                              ▼
            learner.py   experience bandit per (situation, mode) + logistic fall-risk model
                         on sensor features and Jev answers -> runs/jev_agent/<tag>/learner.json
```

**Why this split.** Jev takes text only, takes about 0.3–1 s per call from here, and is weak
at arithmetic (docs: model-jaggedness). So it never touches joints. It makes the
judgment calls (situation plus mission, then gait), all arithmetic and safety limits
stay in code, and the PPO policy does the balancing. Jev's weights can't be
fine-tuned, so the self-learning happens in `learner.py`, on top of Jev's outputs.

## Run

The key is read from `TYPESAFE_API_KEY` or from `TYPE_SAFE_API_KEY` in `.env` (gitignored).

```bash
uv sync --extra train --extra agent
export PYTHONPATH=.
P=runs/g1-steplength-v1/run/params.pkl   # the trained policy from Kaggle

uv run python -m jev_agent.run $P --brain offline --episodes 3   # dry run, no API calls
uv run python -m jev_agent.run $P --episodes 12                  # Jev agent; learns across runs
uv run python -m jev_agent.run $P --baseline normal --episodes 12  # fixed-gait comparison
uv run python -m jev_agent.run $P --scenario storm --mission "Carry a fragile box carefully." --video results/videos/jev_storm.mp4
```

### Live preview

```bash
uv run python -m jev_agent.live $P --port 8765     # then open http://localhost:8765
```

This shows the G1 rendered live from MuJoCo in real time, next to Jev's answer (gait
probabilities, confidence, stability, the yes/no answers), what Jev was told, the raw sensors,
the decision log and past episodes. Jev is called asynchronously, as it would be on hardware,
and the tilt safety filter runs every tick. Change the scenario or mission from the page; it
applies from the next episode. The learner persists to `runs/jev_agent/live/learner.json`.
In the Claude desktop app it's the `jev-live` entry in `.claude/launch.json`.

Scenarios: `nominal, slippery, pushes, payload, delay, storm`. Outputs go to
`runs/jev_agent/<tag>/`: `episodes.jsonl` (fell, distance, tracking error, modes used),
`decisions.jsonl` (every state, Jev answer, choice, reason, outcome) and `learner.json`.

## Stairs world

```bash
uv run python -m jev_agent.stairs --episodes 8 --rises 0.04,0.08      # learn across episodes
uv run python -m jev_agent.live --world stairs --port 8766            # live preview (jev-stairs)
```

A staircase (flat approach, N steps up, landing, N steps down) is added in front of Unitree's
pre-trained G1 policy (`g1pipe/sim.py`, which walks reliably; the PPO checkpoints don't yet).

* **Detect (code):** a height scan casts 58 downward rays over 0.15–3 m ahead, like a depth
  camera or LiDAR heightmap. It finds step edges and describes them in words for Jev:
  "stairs going up about one metre ahead: a flight of several steps, low steps (about 5 to 8 cm)".
* **Intent (Jev):** `stairs_action` = no_stairs / climb / stop_before, based on the mission.
* **Reflex (code, 10 Hz, fresh scan):** stop 0.7 m before the edge, or commit to a climbing gait
  from 1.4 m and hold it until off the stairs. Runs without waiting for Jev, which matters because
  Jev sometimes takes 1–2 s to answer.
* **Skill (learned):** per step size, crossed/failed counts for each gait. It explores untried gaits,
  and once every gait has failed at least twice (< 30 % success) it stops in front of those stairs.

What the robot can physically do is set by the walking policy, not the agent. Unitree's
blind, flat-ground policy crosses 4 cm steps with long fast strides (0.7 m/s, 1.0 s period;
slow short steps stub the toe) and falls on 6 cm and higher with every gait we swept. After about
16 episodes the agent had learned exactly that: stride on low steps, stop before anything taller.
Real stair climbing needs a policy trained on stairs with the height scan in its observation.

## Simulated head camera (vision)

```bash
uv run python -m jev_agent.vision_check --rise 0.08      # figure + metrics in results/vision/
uv run python -m jev_agent.stairs --terrain camera ...   # stair detector reads the camera map
```

A D435i-like depth camera sits on the head: 0.44 m above the pelvis, pitched 45° down,
58° vertical field of view, 0.3–3 m range, 212×120 depth image. Depth noise grows with
distance squared and 3 % of pixels drop out. Pixels on the robot's own body (the hands are in
view) are removed. Points go into a 5 cm world-frame elevation map (per-cell mean, smoothed
over frames), and the profile ahead is read from that map. Since the simulator knows the true
terrain, every frame is scored.

* Accuracy on 8 cm and 12 cm stairs: about 0.5–1 cm mean height error, and 2–5 cm at the 95th
  percentile, all of it at step edges. Coverage of the policy's scan patch reaches 100 % about
  2 m before the stairs.
* Blind spots it shows: the top of a staircase is hidden behind its own steps until the robot
  is close, and the ground under and behind the robot is only known from earlier frames.
* What vision broke, and the fix: the camera blurs each step edge over 2–3 samples, so the
  old jump detector under-measured 8 cm steps as low and the agent tried to climb them. Edges
  are now found as level changes between flat treads, so camera and raycast measure the same
  step heights (within 1 mm) and make the same decisions in the regression.
* The live preview (`--world stairs`) shows the head RGB, the depth image and camera-vs-truth
  profiles, and runs detection from the camera by default (`--terrain raycast` to compare).

## Files

| File | What |
|---|---|
| `questions.py` | **All questions, gait table and thresholds.** Review and tune here. |
| `perception.py` | Robot adapters (`PPORobot`, `VendorRobot`), per-tick sensor aggregation, number-to-word bucketing |
| `brain.py` | `JevBrain` (TypeSafe SDK, pinned model, short retries) and `OfflineBrain` (rule stub for dry runs, not Jev) |
| `supervisor.py` | Decision loop; plugs into `PolicyRunner.run(schedule=...)` with no change to `g1pipe` |
| `learner.py` | Experience bandit, fall-risk logistic regression, and the stairs skill memory, persisted as JSON |
| `run.py` | Scenario × mission episodes, baselines, logging, learning |
| `live.py`, `live.html` | Real-time preview server (MJPEG stream + dashboard), `--world flat|stairs` |
| `vision.py`, `vision_check.py` | Simulated head depth camera, elevation map, scoring against simulator truth |
| `stairs.py` | Staircase scene, height-scan sensor and edge analysis, Unitree-policy episode runner, CLI |

## Status and next steps

* Tested end to end against the live API on the 6M-step probe checkpoint. That policy
  falls after about 1.2 s under any command, so **agent-vs-baseline numbers need the full
  Kaggle run**.
* `run.py` blocks on each Jev call so results are reproducible. `Supervisor(async_brain=True)`
  (used by the live preview) is the hardware-style mode.
* The gait modes are the policy's trained command range. Retune `GAIT_MODES` once
  the E2 tracking grid shows where step-length tracking is accurate.
* Once a camera or heightmap is added, bucket terrain in code (for example "step up ahead")
  and add a `terrain` Choice. Jev still sees only text.
