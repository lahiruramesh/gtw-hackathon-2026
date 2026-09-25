# Jev agent: 2-minute demo script (Gbg Tech Week, SKF track)

Every number here comes from a file in this repo, and the file is cited next to it. Where a number
comes from a checker's run log and not a results file, it says so. All numbers were re-derived from the raw
files on 2026-09-23.

## 1. Pitch (one sentence)

A Unitree G1 humanoid that describes its own sensors in words, asks TypeSafe's Jev model what to do
next as typed questions, keeps every safety limit in code, and learns from its own falls. For example,
it learns that 8 cm stairs are too tall for its walking policy and stops in front of them.

## 2. Setup checklist (T-15 min)

- [ ] `cd /Users/lahiruramesh/sylonik/gtw && uv sync --extra train --extra agent`
- [ ] Check that `.env` has `TYPE_SAFE_API_KEY`. Do not open or show the file on screen.
- [ ] Check that the flat policy exists at `runs/g1-steplength-v1/run/params.pkl`.
- [ ] Start both previews. In the Claude desktop app, use the `.claude/launch.json` entries `jev-live` and `jev-stairs`. Or run them in terminals:
  ```bash
  PYTHONPATH=. uv run python -m jev_agent.live runs/g1-steplength-v1/run/params.pkl --port 8765   # flat world, v1 PPO policy
  PYTHONPATH=. uv run python -m jev_agent.live --world stairs --port 8766                         # stairs, Unitree policy, head camera
  ```
  Open http://localhost:8766 (the main demo) and http://localhost:8765 (backup).
- [ ] On the stairs page, the brain pill should read Jev and the latency line should show about 0.3-0.5 s. The stairs page uses the learner at `runs/jev_agent/live_stairs/learner.json`, which already contains the learned stairs skill. Do not delete it before the demo.
- [ ] **Offline fallback.** If venue Wi-Fi fails, the agent switches by itself after 3 consecutive errors: the pill turns red and reads "OFFLINE MODE (Jev unreachable)". To start in offline mode on purpose:
  ```bash
  PYTHONPATH=. uv run python -m jev_agent.live --world stairs --brain offline --port 8766
  PYTHONPATH=. uv run python -m jev_agent.live runs/g1-steplength-v1/run/params.pkl --brain offline --port 8765
  ```
  Say this plainly on stage: "offline mode is a rule stub, not Jev."
- [ ] Have `results/vision/camera_check_8cm.png` and `results/bench/agent_vs_fixed.md` open in tabs.

## 3. Beat sheet (2:00)

| time | on screen | say |
|---|---|---|
| 0:00-0:15 | stairs page, robot walking | "This is a G1 humanoid in MuJoCo. A neural policy does the balancing at 50 Hz. Above it sits an agent that turns sensors into words, asks Jev what to do, and learns from what happens." |
| 0:15-0:45 | Scenario **4 cm steps**, Mission A "Walk to the stairs, climb them and continue across to the other side." Then Mission B "Walk to the stairs and wait there. Do not climb them." | "Same stairs, two missions. For A, Jev answers `stairs_action = climb`, and once the edge is inside 1.4 m the code commits to the long-stride gait and crosses. For B, Jev answers `stop_before`, and a 10 Hz reflex in code stops the robot once the first edge is within 0.7 m. The reflex doesn't wait for Jev." (Recorded runs, raycast terrain: A crossed 4 cm steps, stride committed at a logged edge distance of 0.95 m, `max_x` 10.04 m. B stopped before, `max_x` 2.08 m, about 0.4 m short of the first step at x = 2.5 m. Both in `runs/jev_agent/stairs_jev/episodes.jsonl` and `decisions.jsonl`.) |
| 0:45-1:05 | Scenario **8 cm steps**, Mission A. Point at the "Learned stairs skill" panel. | "Now the mission says climb, but the robot stops anyway, and the reason reads `learned_too_hard`. Nobody hard-coded that 8 cm is too tall. In earlier training episodes (run with the offline stub brain) it fell on mid-height steps with every gait: cautious 0 of 3 crossed, normal 0 of 3, stride 0 of 2. Once every gait has failed at least twice, the skill memory says stop." (`runs/jev_agent/stairs_jev/learner.json`, `stairs.mid_steps`; the live page's learner `runs/jev_agent/live_stairs/learner.json` has the same mid_steps counts, but more low_steps and tall_steps entries, so the panel will not match table 5B exactly.) |
| 1:05-1:25 | "Head depth camera vs. simulator truth" panel | "On this page, step heights come from a simulated head depth camera turned into an elevation map, and we scored it against the simulator's true terrain. On 8 cm stairs the mean height error along the walking line is about 0.6 cm. The camera is for the stair detector. The walking policy itself is blind, and Jev never sees the image, only words like 'very low steps'." (`results/vision/camera_check_8cm.json`, 30 frames) |
| 1:25-1:45 | "What Jev sees" and "Jev's answer" panels | "This is the actual request: plain text, words like 'feet firmly planted', a terrain sentence, and the mission. No images. The answer is typed: a gait Choice with probabilities, a stability Score, and yes/no Nouls, each with a confidence. Jev makes the judgment call. Code owns the arithmetic and the safety limits, and can override Jev." |
| 1:45-2:00 | `results/bench/agent_vs_fixed.md` | "Does it help? 18 episodes per agent on the trained v1 policy: nominal, slippery, pushes, payload, delay, storm, each with a 'hurry', a 'fragile' and a 'no rush' mission. Outside the delay and storm cases, which knock over every agent, the Jev agent never fell, 0 of 12. On hurry missions it covered 1.6 times the distance of the safest fixed gait, 4.33 m vs 2.70 m. Fixed normal and stride fell on every slippery floor; Jev didn't. The safest fixed gait still has one fewer fall overall, 5 vs 6." |

## 4. Q&A prep: honest limits

- **How tall a step can it climb?** About 4 cm. Unitree's blind, flat-ground policy crosses 4 cm steps with fast long strides and falls on 6 cm and higher with every gait we swept (`jev_agent/README.md`, "Stairs world"). In the learning run (offline stub brain, 17 episodes) it fell in all 8 attempts at 8 cm, then stopped before the stairs in the last 2 episodes. It also fell once at 4 cm with the normal gait and crossed 6 of 7 times at 4 cm (`runs/jev_agent/stairs_offline_learn/episodes.jsonl`). The agent decides whether to climb. The walking policy sets what is physically possible.
- **A stairs-trained policy?** Yes, but the first one failed. Stairs v1 (PPO from scratch, all step heights mixed from the start, 200M steps) fell on 3 of 5 runs even at 3 cm and never reached the top at 9 cm or higher (`results/stairs/eval_v1.json`). On the held-out test stairs it crossed 1 of 32 (`results/stairs/eval_v1_heldout.json`). That is worse than Unitree's blind policy at 4 cm. Stairs v2 fixes how it was trained: a terrain curriculum (start on 2-6 cm, move up a level after crossing a staircase, down after a fall) and a warm start from the flat v1 policy. It is training on Kaggle (`g1-stairs-v2`), and results go to `results/stairs/eval_v2.json`.
- **Jev latency?** On the 147 real calls in `runs/jev_agent/stairs_jev/decisions.jsonl`, the median is 389 ms and p95 is 2065 ms (max 5789 ms). In the benchmark (`wf_bench_jev`, 234 calls), the median is 370 ms and p95 is 784 ms. One run that ran alongside 8 others (`wf_stairs_9`, concurrency per checker log) had a median of 2003 ms and p95 of 3896 ms, so latency under load is real. That is why the stairs reflex and the tilt filter run in code at 10 Hz and every tick, and why Jev is called asynchronously in the live preview.
- **What happens without network?** Each request times out at 4 s (`jev_agent/brain.py`; Jev latency varied from about 0.4 s to 2 s over the day, so 2 s caused false timeouts). After 3 consecutive errors the agent switches to the offline stub, shows it on the dashboard, and retries Jev every 15 s or every 20 decisions. Unknown stairs intent defaults to stopping. In the offline check with the "wait" mission, the robot stopped before the stairs with no fall (`runs/jev_agent/wf_netoff_stairs/episodes.jsonl`). Before the fix, the same setup walked onto the stairs and fell (checker log).
- **What is Jev, what is code, what is learned?**
  - *Jev:* judgment from words. It picks the gait (Choice), rates stability (Score), answers the slipping, disturbed, mission-speed and mission-care questions (Nouls), and gives the stairs intent (climb, stop_before, no_stairs).
  - *Code:* perception buckets, the safety filter (tilt > 25° or stability < 0.75 means stop), confidence gating (slowing down is always allowed; speeding up is one level at a time and needs gait confidence ≥ 0.6 or a positive learned bonus), the stairs reflex, and the offline fallback.
  - *Learned:* a per-situation gait bandit, a logistic fall-risk model, and the stairs skill memory (crossed and failed counts per step size and gait), all stored in `learner.json`.
  - Jev's weights are never fine-tuned.
- **Why does fixed cautious still have fewer falls?** One extra fall, in storm (slippery + pushes + payload), where every agent fell in at least 2 of 3 episodes. Cautious also never goes fast: on hurry missions it covers 38% less distance than the Jev agent (2.70 vs 4.33 m); on fragile and no-rush missions they are about equal (2.67 vs 2.48 m, 2.75 vs 2.75 m). Outside delay and storm, the Jev agent had 0/12 falls (`results/bench/agent_vs_fixed.md`).
- **What changed since the first benchmark?** The first run (12 episodes) had Jev tie cautious at 3/12 and choose cautious 216 of 234 times. A slip detector counted normal heel-strike as slipping, so Jev was told "feet sliding badly" constantly. Fixes: slip is now measured at loaded contact points and must last 60 ms, descriptions are calibrated to each robot's normal walking (`jev_agent/calibration.json`), slip is remembered for 2 s, and the fastest gait needs 1.5 s of clean footing.
- **Sample size?** 18 episodes per agent, 3 per scenario, so a one-fall difference is noise. Every scenario now meets every mission (the first benchmark confounded them).

## 5. Results (one page)

### A. Agent vs fixed gaits, flat world, v1 PPO policy, 18 episodes x 12 s each (`results/bench/agent_vs_fixed.md`, `.json`)

| agent | falls (18) | falls excl. delay (15) | mean distance | hurry (6) | fragile (6) | no rush (6) | gait decisions |
|---|---|---|---|---|---|---|---|
| fixed cautious | 5/18 | 2/15 | 2.71 m | 1/6 falls, 2.70 m | 2/6 falls, 2.67 m | 2/6 falls, 2.75 m | - |
| fixed normal | 9/18 | 6/15 | 3.43 m | 3/6 falls, 3.38 m | 3/6 falls, 3.29 m | 3/6 falls, 3.61 m | - |
| fixed stride | 8/18 | 5/15 | 6.13 m | 2/6 falls, 6.40 m | 3/6 falls, 6.18 m | 3/6 falls, 5.82 m | - |
| offline-brain agent | 7/18 | 4/15 | 2.79 m | 2/6 falls, 2.34 m | 2/6 falls, 2.56 m | 3/6 falls, 3.46 m | cautious 221, normal 62, stop 30 |
| Jev agent | 6/18 | 3/15 | 3.19 m | 2/6 falls, 4.33 m | 2/6 falls, 2.48 m | 2/6 falls, 2.75 m | cautious 252, stride 35, normal 28, stop 12 |

| scenario | fixed cautious | fixed normal | fixed stride | offline-brain agent | Jev agent |
|---|---|---|---|---|---|
| nominal | 0/3, 3.3 m | 0/3, 5.5 m | 0/3, 9.5 m | 0/3, 3.9 m | 0/3, 4.2 m |
| slippery | 0/3, 3.6 m | 3/3, 1.3 m | 3/3, 1.4 m | 0/3, 3.8 m | 0/3, 3.9 m |
| pushes | 0/3, 3.1 m | 0/3, 5.5 m | 0/3, 9.0 m | 1/3, 3.1 m | 0/3, 3.5 m |
| payload | 0/3, 3.1 m | 0/3, 5.7 m | 0/3, 9.0 m | 0/3, 3.9 m | 0/3, 4.6 m |
| delay | 3/3, 0.8 m | 3/3, 0.6 m | 3/3, 1.4 m | 3/3, 0.6 m | 3/3, 0.7 m |
| storm | 2/3, 2.4 m | 3/3, 2.0 m | 2/3, 6.5 m | 3/3, 1.4 m | 3/3, 2.1 m |

Jev agent: 327 decisions answered by jev-1.13.0, 2 API errors (fell back to cautious for that decision), latency median 394 ms, p95 585 ms. Reasons: {'keep': 214, 'hold_low_confidence': 56, 'speed_up_one_level': 24, 'hold_until_ground_proven': 14, 'slow_down': 12, 'risk_veto': 4}

### B. Stairs world, Unitree policy (`runs/jev_agent/<tag>/episodes.jsonl`)

| run | brain | terrain | rise | mission | outcome |
|---|---|---|---|---|---|
| stairs_jev ep0 | Jev | raycast | 4 cm | climb | crossed, stride, no fall, max_x 10.04 m |
| stairs_jev ep1 | Jev | raycast | 8 cm | climb | stopped before (`learned_too_hard` ×36), max_x 2.08 m |
| stairs_jev ep2 | Jev | raycast | 4 cm | wait | stopped before (`stop_before_stairs` ×36), max_x 2.08 m |
| wf_stairs_5 | offline | camera | 4 cm | climb | crossed, stride, max_x 12.99 m |
| wf_stairs_6 | offline | camera | 5 cm | climb | stopped before (`learned_too_hard` ×41) |
| wf_stairs_9 | Jev | camera | 8 cm | climb | stopped before (`learned_too_hard` ×36), 0 brain errors |
| wf_stairs_climb_jev | Jev | raycast | 4 cm | climb | crossed, stride, max_x 10.93 m, 0 brain errors |
| wf_netoff_stairs | API unreachable | camera | 4 cm | wait | stopped before, no fall, 2 brain errors, then offline |

Every one of the 9 `wf_stairs_1`..`wf_stairs_9` regression runs (1-4 raycast, 5-9 camera; 4/5/8 cm; climb and wait; 1-8 offline stub, 9 Jev) had the expected outcome with no falls: 4 cm climb crossed, 5 and 8 cm climb stopped (`learned_too_hard`), 4 cm wait stopped (`stop_before_stairs`) (each run's `episodes.jsonl`). The "stairs_jev" rows used raycast terrain (older log without a `source` field; raycast is the `stairs.py` default).

**Learned stairs skill** (`runs/jev_agent/stairs_jev/learner.json`, [crossed, failed]):

| gait | low_steps | mid_steps |
|---|---|---|
| cautious | 0, 0 | 0, 3 |
| normal | 0, 1 | 0, 3 |
| stride | 7, 0 | 0, 2 |

### C. Head camera vs simulator truth (30 frames each, from 2.49 m to 0.54 m before the stairs)

| stairs | mean height err, line (cm) | worst frame mean (cm) | worst p95 (cm) | line coverage | patch mean err (cm) | patch coverage | file |
|---|---|---|---|---|---|---|---|
| 8 cm | 0.57 | 0.78 | 4.36 | 90-100 % | 0.18 | 60-100 % | `results/vision/camera_check_8cm.json` |
| 12 cm | 0.92 | 1.30 | 7.12 | 84-100 % | 0.24 | 60-100 % | `results/vision/camera_check_12cm.json` |

The patch is the 11 x 5 height scan a stairs-PPO policy would observe (`jev_agent/vision_check.py`). No trained policy uses it yet: the Unitree walking policy in the demo is blind, and the camera only feeds the stair detector, which turns heights into words for Jev.

### D. Jev API (computed from `jev.latency_s` in the decision logs, error-free Jev calls only)

| log | calls | median (ms) | p95 (ms) | errors |
|---|---|---|---|---|
| `runs/jev_agent/stairs_jev/decisions.jsonl` | 147 | 389 | 2065 | 0 |
| `runs/jev_agent/wf_bench_jev/decisions.jsonl` | 234 | 370 | 784 | 0 |
| `runs/jev_agent/wf_stairs_climb_jev/decisions.jsonl` | 49 | 359 | 445 | 0 |
| all Jev calls in `stairs_jev` and `wf_*` logs | 552 | 379 | 2059 | - |

All successful Jev calls used model `jev-1.13.0`. Errors elsewhere: 1 HTTP 529 in `wf_flat_jev_fix`, 1 timeout in `wf_flat_jev_speed`, and the deliberate network-off runs (`wf_netoff_*`).
