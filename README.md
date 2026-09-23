# G1 step-length pipeline — SKF track, Gbg Tech Week × Chalmers

A practical learning and validation pipeline for a humanoid locomotion task:
**commanding the Unitree G1's step length**. Everything except GPU training
runs on a laptop CPU (developed on an Apple M4, no GPU). Training runs on a
free Kaggle T4 GPU and is launched from the laptop.

```
spec ─► sim env ─► rewards + randomisation ─► train (GPU, Kaggle) ─► evaluate (CPU, other engine) ─► stress tests ─► release gate
         g1pipe/steplength_env.py        g1pipe/train.py            g1pipe/evaluate.py  scripts/eval_suite.py
```

## Quick start

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym third_party/unitree_rl_gym
uv sync --extra train
export PYTHONPATH=.
```

| Step | Command | Where |
|---|---|---|
| Vendor policy walks + video | `uv run python scripts/run_policy.py --vx 0.5 --video results/videos/baseline.mp4` | Mac |
| **E5** does step length need retraining? | `uv run python scripts/e5_sweep.py` | Mac, ~15 s |
| Pipeline smoke test (tiny training) | `uv run python -m g1pipe.train --smoke --out runs/smoke` | Mac, ~90 s |
| Train on GPU | `uv run python scripts/kaggle_job.py push --name g1-steplength-v1 --timesteps 200000000` (add `--extra="--no-dr"` for E3) | Kaggle T4 |
| Check / fetch | `uv run python scripts/kaggle_job.py status --name g1-steplength-v1` then `pull` | Mac |
| **E2/E3** evaluate + stress + demo video | `uv run python scripts/eval_suite.py runs/g1-steplength-v1/run/params.pkl --tag v1` | Mac |

Kaggle needs `~/.kaggle/kaggle.json` (Kaggle → Settings → API → Create token) and a phone-verified account (for GPU + internet).

## Layout

| Path | What |
|---|---|
| `g1pipe/sim.py` | Headless MuJoCo runner for Unitree's pre-trained 12-DoF G1 policy (LSTM), gait metrics, perturbations, video |
| `g1pipe/steplength_env.py` | Task definition: Playground G1 joystick env + step-length command in obs + touchdown reward |
| `g1pipe/train.py` | Brax PPO training (Playground's tuned G1 recipe), periodic checkpoints, progress CSV |
| `g1pipe/evaluate.py` | Runs a trained policy in plain MuJoCo (C engine) — the cross-engine check |
| `scripts/e5_sweep.py` | E5: step-length range of the vendor policy without retraining |
| `scripts/eval_suite.py` | E2 tracking grid, E3 stress tests, on-the-fly step change demo |
| `scripts/kaggle_job.py` | Bundle + push training to a private Kaggle GPU kernel, poll, pull results |
| `docs/effort_log.csv` | Hours per stage — evidence for "how much effort is required?" |

## Task design (why it looks like this)

* **Command = (speed, step length).** Cadence follows: `f = |v| / (2·ℓ)`. The policy gets `ℓ` and `f`
  as inputs and a gait-phase clock running at `f`. Speed and step length are then independently
  controllable within `f ∈ [0.9, 1.8] Hz`.
* **Reward:** Playground's tuned G1 locomotion terms, plus one new term: at each foot touchdown,
  `exp(-(Δx − ℓ)² / 0.05²)` where `Δx` is how far the landing foot is ahead of the stance foot.
* **Randomisation:** friction, joint friction, armature, link masses, torso payload, joint offsets,
  pushes, sensor noise (Playground defaults). `--no-dr` switches them off for experiment E3.

## Pinned versions

`jax 0.7.2`, `brax 0.14.2`, `playground 0.2.0`, `mujoco 3.14.0`. JAX ≥ 0.8 breaks Brax 0.14 (`device_put_replicated` removed).
