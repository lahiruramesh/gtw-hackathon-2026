# Humanoid skill training pipeline and SKF Skill Studio

SKF track, Gothenburg Tech Week × Chalmers. Team: Lahiru Ramesh, Sophearoth Prum, Muhammad Ahmad Amjad.

This repository trains and validates skills for the **Unitree G1 humanoid** in simulation and wraps the
pipeline in a web platform, **SKF Skill Studio**, where a team can define, launch, monitor, gate and
approve skills. Three skills are implemented: commanded **step length** on flat ground, **stair
climbing** with a head-camera height scan, and **bearing-ring pick and drop**.

| | |
|---|---|
| **Team report (PDF)** | [docs/report/gtw-chalmers-skf-team-report.pdf](docs/report/gtw-chalmers-skf-team-report.pdf) (editable: [.docx](docs/report/gtw-chalmers-skf-team-report.docx)) |
| **Technical report (HTML)** | [docs/report/index.html](docs/report/index.html) |
| **Dashboard** | [sylonik.com](https://sylonik.com) (accounts are created by an admin; no public sign-up) |
| **Dashboard screenshots** | [docs/report/screenshots/dashboard](docs/report/screenshots/dashboard) |
| **Videos** | [results/videos](results/videos) |

## Latest results (simulation only)

| Skill | Best version | Result | Gate |
|---|---|---|---|
| Stair climbing | `g1-stairs-v14` | Strict test on unseen stairs: 95/96 crossed, 1 fall (v1: 28/32 fell); qualified height 9.9 cm with the true terrain scan, 6.4 cm with the head camera | Simulation gate passed; hardware release gate **not passed** (9.86 cm < 10 cm; camera, 40 ms delay and 5 kg payload still fail) |
| Step length | `g1-steplength-v1` | No falls on the 20-command grid in both engines; 10.7 cm mean step error in the second engine | Simulation gate passed; release failed on step error and latency |
| Ring pick and drop, arm stand-in | `g1-ring-v2` | 1024/1024 successes at 21M steps, no raceway contact | Passes its success criterion |
| Ring pick and drop, G1 + Dex3 hand | `g1-ring-humanoid-v2` | Holds and lifts the ring (56 % of episodes), no completed placement yet | Not passed |

"Qualified" means a policy passed our stated simulation tests. It is an internal gate, not a safety
certification, and nothing here has run on hardware. The stair suites were used during development, so
they are validation results; a frozen, untouched final test is the next milestone. Known caveat: the CPU
evaluator applies one extra control step of latency, so "nominal" tests ran at 20 ms and the delay test at
40 ms (see [docs/report/index.html](docs/report/index.html) §5).

## How the pipeline works

```
skill spec ─► preflight ─► train (GPU) ─► screen checkpoints ─► qualify (second engine) ─► gate + human review
skills/*.yaml  g1pipe/      g1pipe/        g1pipe/gpu_eval.py     g1pipe/stairs_eval.py      Skill Studio
               preflight.py train.py       g1pipe/bench.py        scripts/certify.py
```

* **Train** with Brax PPO on MuJoCo Playground (MuJoCo Warp/MJX on a GPU, 8,192 robots in parallel),
  with a terrain curriculum, warm starts from the closest skill and targeted domain randomisation.
* **Test** in plain MuJoCo (C engine) on a CPU, a different contact implementation, on staircases the
  policy never trained on, with confidence intervals on every rate.
* **Release** only when every gate criterion passes and a safety reviewer who did not train the policy
  approves it.

## Getting started

Requirements: Python 3.11 with [uv](https://docs.astral.sh/uv/). Everything except GPU training runs on a
laptop CPU (developed on an Apple M4).

```bash
git clone https://github.com/lahiruramesh/gtw-hackathon-2026.git && cd gtw-hackathon-2026
git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym third_party/unitree_rl_gym
uv sync --extra train
export PYTHONPATH=.
```

### Try it on a laptop (no GPU)

| What | Command | Time |
|---|---|---|
| Vendor policy walks, with video | `uv run python scripts/run_policy.py --vx 0.5 --video results/videos/baseline.mp4` | ~1 min |
| Baseline sweep: step length without training (E5) | `uv run python scripts/e5_sweep.py` | ~15 s |
| Pipeline smoke test (tiny training run) | `uv run python -m g1pipe.train --smoke --out runs/smoke` | ~90 s |
| Watch a trained policy live (3D viewer, keyboard control) | `scripts/view.sh --run v1` | — |
| Strict stairs test of the best policy | `uv run python -m g1pipe.stairs_eval runs/g1-stairs-v14/run/params.pkl --vx 0.7 --starts 3` | ~20 min |
| Policy card (8-condition qualification suite) | `uv run python scripts/certify.py runs/g1-stairs-v14/run/params.pkl --target-cm 10` | ~20 min |

### Train on a GPU

One command starts a GPU machine (AWS g6e / L40S, with zone fallback), runs the preflight, trains, ranks
every checkpoint and stops the machine. A daily time cap and a shutdown timer on the machine itself are
always set.

```bash
uv run python scripts/g1job.py --run g1-stairs-v15 --init-from runs/g1-stairs-v14/run/params.pkl --budget-min 90 -- \
  --task stairs --leg-action-scale 1.0 --lr 1e-4 --timesteps 300000000 --scan-model camera
```

| Step | Command |
|---|---|
| Preflight only (engine parity, throughput, warm start) | `uv run python -m g1pipe.preflight --init-from <params> --leg-action-scale 1.0` |
| Rank every checkpoint (2,048 crossings each, 95 % intervals) | `uv run python -m g1pipe.gpu_eval runs/<run>/run/ckpt_*.pkl --scan camera` |
| Fixed 7-condition benchmark | `uv run python -m g1pipe.bench runs/<run>/run/params.pkl --out bench.json` |
| Ring pick and drop, arm stand-in / G1 + Dex3 | `--task ring` / `--task g1ring`, evaluated with `uv run python -m g1pipe.ring_eval <params> --episodes 1024` |
| Kaggle T4 alternative | `uv run python scripts/kaggle_job.py push --name <run> --timesteps 200000000 --extra="--task stairs"` |
| Box helper | `scripts/aws_box.sh status / start / stop / ssh / sync / train` |

## SKF Skill Studio (web platform)

Sign in, launch training on the local CPU or a cloud GPU, watch logs and metrics live, run evaluation and
the two gate levels (simulation gate, hardware release gate) automatically, compare runs, and approve
releases by role (admin, ML engineer, operator, safety reviewer, viewer). Stack: Next.js web, FastAPI API
and worker, Postgres, Redis and S3.

```bash
make infra-up api-install web-install migrate seed   # once (copy the .env.example files first)
make api      # API on :8000
make worker   # background worker
make web      # http://localhost:3100
make import-history HISTORY_DIR=runs RESULTS_DIR=results   # optional: import past runs
make test lint
```

| Doc | What |
|---|---|
| [docs/webapp/PLAN.md](docs/webapp/PLAN.md) | Why the app exists and how it is designed |
| [docs/webapp/SPEC.md](docs/webapp/SPEC.md) | API, data model, run state machine, compute backends, UI, manifest schema (§7) |
| [docs/webapp/RUNBOOK.md](docs/webapp/RUNBOOK.md) | Local setup, users and roles, compute targets, troubleshooting |
| [infra/aws/README.md](infra/aws/README.md) | Production deployment on AWS (EC2, RDS, S3, Caddy TLS) |

### Define a new skill

A skill is a YAML manifest in `skills/<name>/skill.yaml` plus a small `summarize.py` that turns each
stage's output into gate metrics. The platform validates the manifest and generates the forms, pipeline
and gates from it (**Skills → Sync from repository**).

```yaml
id: g1-stairs
robot: unitree-g1
category: locomotion
method: rl-ppo-curriculum
params:
  timesteps: {type: integer, default: 200000000}
  init_from: {type: checkpoint, nullable: true}      # warm start from another skill
pipeline:
  - {id: train, kind: train, runs_on: target,
     argv: [python, -m, g1pipe.train, --task, stairs, --out, "{out_dir}"]}
  - {id: evaluate, kind: evaluate, runs_on: local,
     argv: [python, -m, g1pipe.stairs_eval, "{input_dir}/train/params.pkl", --starts, "3"]}
  - {id: gate, kind: gate}
gate:
  - {metric: evaluate.crossed_rate, op: ">=", value: 0.9}
  - {metric: evaluate.certified_cm, op: ">=", value: 10}
```

Existing skills: [`step_length`](skills/step_length/skill.yaml), [`stairs`](skills/stairs/skill.yaml),
[`stairs_bench`](skills/stairs_bench/skill.yaml) (one-change experiments on a fixed benchmark) and
[`ring_pick`](skills/ring_pick/skill.yaml). A new variant of a supported task is just a new manifest; a new
task also needs an environment and an evaluator in `g1pipe/`.

## Repository layout

| Path | What |
|---|---|
| `g1pipe/` | Environments (`steplength_env.py`, `stairs_env.py`, `stairs_terrain.py`, `ring_env.py`, `g1_ring_env.py`), training (`train.py`), evaluation (`evaluate.py`, `stairs_eval.py`, `gpu_eval.py`, `bench.py`, `ring_eval.py`), `preflight.py`, and the improvement loop (`improve/`) |
| `scripts/` | Job runners (`g1job.py`, `kaggle_job.py`, `aws_box.sh`), `certify.py`, `eval_suite.py`, `e5_sweep.py`, videos and viewers |
| `skills/` | Skill manifests for the platform |
| `apps/` | Skill Studio: `api/` (FastAPI + worker), `web/` (Next.js), `reporter/` (in-job log and metric reporter) |
| `infra/` | Docker Compose and AWS deployment |
| `jev_agent/` | Optional supervisor agent that picks the gait from sensors and mission ([README](jev_agent/README.md)) |
| `experiments/` | Improvement-campaign definitions |
| `results/` | Result files, policy cards, figures and videos behind every number in the reports |
| `runs/` | Checkpoints and training logs per run |
| `docs/` | Reports, [pipeline notes](docs/pipeline.md), [lessons as rules](docs/lessons.md), [improvement loop](docs/improve.md), [effort log](docs/effort_log.csv) |
| `tests/` | Pipeline tests (`uv run pytest tests`) |

## Pinned versions

`jax 0.7.2`, `brax 0.14.2`, `playground 0.2.0`, `mujoco 3.14.0`. JAX ≥ 0.8 breaks Brax 0.14
(`device_put_replicated` was removed).
