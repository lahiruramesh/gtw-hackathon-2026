# Jev agent vs fixed gaits: trained v1 PPO policy, flat world

18 episodes per agent, 12 s each, identical seeds. Every scenario (nominal, slippery ×0.35 friction,
pushes, +6 kg payload, 40 ms actuation delay, storm = slippery + pushes + payload) meets every
mission ("hurry", "carry a fragile box carefully", "no rush"), 3 episodes per scenario.
The agents start with an empty learner and learn across the 18 episodes; the fixed gaits don't learn.
Run after the perception fixes (load-gated sustained slip, per-robot calibration, 2 s slip memory)
and the "earn the fastest gait" rule (1.5 s of clean footing before stride).

Raw data: `runs/jev_agent/{wf2_base_cautious,wf2_base_normal,wf2_base_stride,wf2_offline_v3,wf2_jev_v3}/episodes.jsonl`
(plus `decisions.jsonl` for the agents). Reproduce: `python -m jev_agent.run runs/g1-steplength-v1/run/params.pkl
--episodes 18 --duration 12 [--baseline cautious|normal|stride | --brain jev|offline] --tag <tag>`.

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

## Reading it

* Outside delay and storm, the Jev agent never fell (0/12). Every agent fell in all 3 delay episodes,
  and storm knocked over every agent in 2–3 of its 3 episodes.
* Jev follows the mission. On "hurry" it covered 4.33 m against 2.70 m for fixed cautious (1.6×). On
  "fragile" and "no rush" it walks like cautious (2.48 / 2.75 m).
* Fixed normal and fixed stride fall on every slippery episode (3/3). The Jev agent fell on none.
* Fixed cautious still has the fewest falls overall (5 vs 6): the one extra fall is in storm. Stride
  covers the most distance by ignoring the mission and the floor.
* Offline-brain agent (same code, a hand-written rule stub instead of Jev): 7/18 falls, 2.79 m, and
  2.34 m on hurry. It can't read missions as well as Jev.
* Small sample: 3 episodes per scenario, so a one-fall difference is noise.

## History

The first run of this benchmark (12 episodes, before the fixes) had the Jev agent tie cautious at 3/12
and pick cautious in 216 of 234 decisions. A slip detector that counted normal heel-strike as slipping
told Jev "feet sliding badly" on every window. Fixing that, calibrating the descriptions to each robot's
normal walking, adding slip memory, and requiring clean footing before stride produced the table above.
