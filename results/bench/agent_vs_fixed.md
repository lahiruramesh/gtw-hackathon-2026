# Jev agent vs fixed gaits (trained v1 policy)

Policy: `runs/g1-steplength-v1/run/params.pkl`. 12 episodes x 12 s per agent. Episode i uses scenario SCENARIOS[i % 6], mission MISSIONS[i % 3] and seed i, so every agent sees the same conditions.
Command: `python -m jev_agent.run runs/g1-steplength-v1/run/params.pkl --episodes 12 --duration 12 --tag <TAG> [--baseline M | --brain jev|offline]`.
Tags: wf_bench_base_{cautious,normal,stride}, wf_bench_jev, wf_bench_offline (under runs/jev_agent/). Raw numbers: `results/bench/agent_vs_fixed.json`.

## Headline

The Jev agent tied fixed-cautious on falls (3/12 each), the fewest of any agent, and behaved almost the same as fixed-cautious: 216 of 234 decisions were `cautious`, 18 were `stop`, and it never chose `normal` or `stride`. It **does not beat the best fixed gait**. Its only real difference is on storm, where it stopped more (1.84 m vs 2.55 m in the episode both survived). It beat `normal` and `stride` on falls because both of those fall on every slippery episode, but it covered less than half of stride's distance on the benign scenarios. The offline brain (no API) fell 4/12. The Jev API had 0 errors, with a mean latency of about 450 ms per question.

## Overall

| agent | falls | mean distance (m) | mean vx abs err (m/s) | Jev latency (ms) | brain errors | modes |
|---|---|---|---|---|---|---|
| fixed cautious | 3/12 | 2.721 | 0.108 | - | - | - |
| fixed normal | 6/12 | 3.495 | 0.274 | - | - | - |
| fixed stride | 5/12 | 6.112 | 0.43 | - | - | - |
| Jev agent | 3/12 | 2.615 | 0.102 | 448.725 | 0 | {'cautious': 216, 'stop': 18} |
| offline agent | 4/12 | 2.543 | 0.106 | 0.0 | 0 | {'cautious': 206, 'stop': 21} |

(falls = episodes with a fall; distance and vx error are per-episode means; vx error ignores NaN episodes. Latency and error counts come from the brain's per-decision log. Modes = decision counts over all 12 episodes.)

## Per scenario (2 episodes each)

| scenario | fixed cautious | fixed normal | fixed stride | Jev agent | offline agent |
|---|---|---|---|---|---|
| nominal | 0/2 falls, 3.26 m | 0/2 falls, 5.48 m | 0/2 falls, 9.48 m | 0/2 falls, 3.26 m | 0/2 falls, 3.26 m |
| slippery | 0/2 falls, 3.56 m | 2/2 falls, 1.28 m | 2/2 falls, 1.36 m | 0/2 falls, 3.56 m | 0/2 falls, 3.56 m |
| pushes | 0/2 falls, 3.09 m | 0/2 falls, 5.75 m | 0/2 falls, 9.045 m | 0/2 falls, 3.04 m | 0/2 falls, 2.885 m |
| payload | 0/2 falls, 3.11 m | 0/2 falls, 5.65 m | 0/2 falls, 9.01 m | 0/2 falls, 3.11 m | 0/2 falls, 3.11 m |
| delay | 2/2 falls, 0.75 m | 2/2 falls, 0.56 m | 2/2 falls, 1.44 m | 2/2 falls, 0.52 m | 2/2 falls, 0.81 m |
| storm | 1/2 falls, 2.555 m | 2/2 falls, 2.25 m | 1/2 falls, 6.335 m | 1/2 falls, 2.2 m | 2/2 falls, 1.635 m |
## Why the agent never speeds up

The perception layer reports `foot_grip` = "feet sliding badly" (max_slip > 0.4) or "feet slipping on the ground" in **every** decision window, including nominal ground. For example, nominal had 36 "sliding badly" windows and 10 "slipping" windows. Jev then returns slipping about 0.96 and puts about 0.9 of its gait probability on `cautious` in every scenario. The slip-speed thresholds in `jev_agent/perception.py` (the `foot_grip` buckets 0.08/0.15/0.4) look miscalibrated for this policy's normal gait. Until they are fixed, the agent can't tell benign ground from slippery ground.

## Caveats

- 12 episodes (2 per scenario) is a small sample. A difference of one fall is noise.
- Scenario and mission are confounded: 6 and 3 share a factor, so each scenario only ever gets one mission. nominal/payload always get "dock, no rush", slippery/delay always get "hurry", and pushes/storm always get "fragile box". The agent was never asked to hurry on benign ground, which is the case where it should pick stride.
- Missions differ in intent. A no-rush or fragile mission SHOULD walk slower, so distance is not the only score. Fixed stride's distance lead partly rewards ignoring the mission.
- The learner started empty (a fresh learner.json per tag) and updated after every episode, so later episodes of the jev and offline runs had more learned bias than earlier ones. Fixed baselines don't learn.
- The delay scenario makes every agent fall (2/2 each), so it doesn't separate agents at all.
- The jev and offline runs ran at the same time as the baselines on one machine. Simulation is deterministic per seed, but Jev latency may be inflated by the load.
