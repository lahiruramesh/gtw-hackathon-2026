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
| Train stairs policy (curriculum, warm start from v1) | `uv run python scripts/kaggle_job.py push --name g1-stairs-v2 --timesteps 200000000 --init-kernel g1-steplength-v1 --extra="--task stairs"` | Kaggle T4 |
| Evaluate stairs policy on held-out stairs | `uv run python -m g1pipe.stairs_eval runs/g1-stairs-v2/run/params.pkl --out results/stairs/eval_v2.json --video results/videos/stairs_v2.mp4` | Mac |
| **Watch it walk** (live 3D, keyboard control) | `scripts/view.sh --run v1` (or `--run nodr`) | Mac |
| **E2/E3** evaluate + stress + demo video | `uv run python scripts/eval_suite.py runs/g1-steplength-v1/run/params.pkl --tag v1` | Mac |

Kaggle needs `~/.kaggle/kaggle.json` (Kaggle → Settings → API → Create token) and a phone-verified account (for GPU + internet).

## Web app (SKF Skill Studio)

A web app wraps this pipeline for a team: sign in, launch training on Kaggle, an AWS GPU box or the local CPU,
watch logs and metrics live, run evaluation and the release gate automatically, compare runs and approve
releases by role. Code is in `apps/` (FastAPI API + worker, Next.js web, in-job reporter), skill manifests in
`skills/`, and Docker/AWS deployment in `infra/`.

```bash
make infra-up api-install web-install migrate seed   # once (copy the .env.example files first)
make api   # :8000        make worker        make web   # http://localhost:3100
```

| Doc | What |
|---|---|
| [docs/webapp/PLAN.md](docs/webapp/PLAN.md) | why the app exists and how it is designed |
| [docs/webapp/SPEC.md](docs/webapp/SPEC.md) | the implementation contract: API, data model, state machine, backends, UI |
| [docs/webapp/RUNBOOK.md](docs/webapp/RUNBOOK.md) | local setup, users and roles, compute targets, troubleshooting, secret rotation |
| [infra/aws/README.md](infra/aws/README.md) | production deployment on AWS (EC2 + RDS + S3, Caddy TLS) |

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
| `g1pipe/stairs_terrain.py`, `g1pipe/stairs_env.py` | Stairs task: 8 × 8 curriculum grid (row = level, 2–16 cm steps; pyramids to walk down, pits to climb out of) + 55-point height scan in the observation + per-env terrain curriculum; `train.py --task stairs [--init-from flat params.pkl]` |
| `g1pipe/stairs_eval.py` | Crosses staircases in plain MuJoCo on the held-out `test` layout (unseen step heights and tread); reached-centre / crossed / fell per step height |
| `jev_agent/` | Self-learning high-level agent: TypeSafe Jev picks the gait from sensors + mission, code gates it, learner adapts. See [jev_agent/README.md](jev_agent/README.md) |
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
