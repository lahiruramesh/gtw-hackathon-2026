# SKF Humanoid Skill Studio — web app plan

Status: **proposal, awaiting approval**. Nothing here is implemented yet.

## 1. What problem are we solving

SKF's brief asks one core question: *how can training industrial humanoids be made easier and more
efficient?* Two things are expected from it: an analysis comparing learning methods against industrial
requirements, and one pipeline taken all the way to simulation (step length or stair climbing), with results,
docs, KPIs, and a proposed pipeline for an SKF use case.

This repo already has that pipeline, but it can only be driven from one laptop with CLI scripts:

```
spec ─► sim env ─► reward + DR ─► train (GPU: Kaggle / Colab / AWS) ─► eval (other engine, CPU)
     ─► stress tests / strict checkpoint test ─► release gate ─► (gantry ─► floor ─► shadow)
```

| Today (CLI, one person) | Pain it causes (from `docs/effort_log.csv`) |
|---|---|
| `kaggle_job.py push/status/pull`, `aws_box.sh train`, Colab notebook | Three different launch paths, credentials on one Mac, quota exhausted without warning |
| Logs live in Kaggle UI, `~/NAME.log` over SSH, or a notebook | Warp printed 30 GB of solver noise in 40 min, and there's no live view for others |
| Results are JSON/CSV/MP4 files in `results/` | Comparing v1–v11 is manual, and the gate verdict is prose in `pipeline.md` |
| Warm starts (`--init-from`, `--init-kernel`) done by hand | Lineage (v11 ← v10@254M ← v9 ← v7 ← v5) only exists in the effort log |
| Effort and GPU hours typed into a CSV | Business KPIs (cost per skill, GPU-h per version) aren't computed |

**The web app makes this pipeline a shared, repeatable, auditable tool for SKF staff**: log in, pick a
skill, launch a run on a chosen GPU location, watch logs and metrics live, have evaluation and the release
gate run automatically, compare versions, and approve (or reject) a policy for hardware trials. It does
**not** control a physical robot. It produces a signed release package that SKF's own deployment
toolchain consumes (see §7).

## 2. Users and roles (Better Auth)

| Role | Who at SKF | Can |
|---|---|---|
| `admin` | Platform owner / IT | Everything; users, roles, compute targets, secrets, integrations |
| `ml_engineer` | Robotics/ML engineers | Create and edit skills, pipelines, and presets; launch any run; evaluate checkpoints |
| `operator` | Process specialists, technicians | Launch runs **from approved presets only**, within a GPU-hour budget; view everything |
| `safety_reviewer` | HSE / safety engineering | Review gate reports, **approve or reject releases** (can't launch runs) |
| `viewer` | Management, partners | Read-only: dashboards, KPIs, reports, videos |

These roles are enforced in two places: in FastAPI, which is the source of truth, and in the UI, where controls
a user can't use are hidden and explained. Every mutation is written to an audit log.
A run over the operator's budget, or on a paid target, needs an `ml_engineer` or `admin` to approve it first.

## 3. Architecture

```
Browser ──► Next.js 16 (App Router, shadcn/ui, Better Auth)  ──server-side fetch + JWT──►  FastAPI
               │ /api/auth/*  (sessions, SSO, roles, JWKS)                                  │
               └─ route handler proxy for SSE log streams                                   │
                                                                                            ▼
             PostgreSQL 17 ◄──── SQLAlchemy 2 / Alembic ────  api  ◄──►  Redis (queue + log pub/sub)
             (schemas: auth, app)                                            ▲
                                                                             │ arq jobs
             MinIO / S3 / GCS ◄──── artifacts (params, ckpts, videos) ─── worker (orchestrator)
                                                                             │ ComputeBackend
                          ┌────────────────┬──────────────┬─────────────────┼───────────────┐
                        Local CPU        Kaggle        AWS EC2 GPU      GCP (Vertex AI)   SKF on-prem
                     (smoke, evals)  (T4, free quota)  (L40S box)      (SKF's suggestion)  (SSH/Slurm/K8s)
                          │                │              │                 │                │
                          └──── in-job reporter POSTs logs, metrics, heartbeat ──► /ingest (public, per-run token)
```

**Key decisions**

| Decision | Choice | Why |
|---|---|---|
| Frontend | Next.js 16.x App Router, React Server Components, `proxy.ts` for route guards | Latest stable; server components keep tokens off the browser |
| UI kit | shadcn/ui (Tailwind v4), Recharts via shadcn charts, TanStack Table, `react-hook-form` + `zod` | Consistent, accessible, owned in-repo |
| Auth | Better Auth 1.7: email/password, Microsoft Entra ID SSO, `admin` plugin with custom access control, `jwt` plugin (JWKS), `twoFactor`, `apiKey` | One auth system; FastAPI verifies JWTs statelessly via JWKS |
| Auth storage | Better Auth's own Postgres adapter in schema `auth`, migrated by its CLI | No second ORM in the web app |
| API | FastAPI 0.14x, async SQLAlchemy 2.1, Alembic, Pydantic v2 | Python side can import skill manifests and reuse `g1pipe` metadata |
| API contract | OpenAPI → `openapi-typescript` + `openapi-fetch` generated client; CI fails on drift | Typed end to end, no hand-written fetch code |
| Jobs | `arq` (async, Redis) worker; durable state machine in Postgres | Runs last hours; state must survive worker restarts. Temporal is overkill at this scale |
| Live logs | Reporter → `/ingest` → Postgres (chunks) + Redis pub/sub → SSE | Works for every backend with outbound internet; falls back to polling |
| Artifacts | S3 API (MinIO in dev; S3/GCS/SKF storage in prod), presigned URLs | Videos stream in browser, checkpoints downloadable |
| Local dev / deploy | Docker Compose (postgres, redis, minio, api, worker, web, caddy) | One command up; same images in prod |

## 4. Core domain model (schema `app`)

```
skill ──< skill_version (git sha, manifest snapshot)
  │
  ├──< preset (named, approved param sets: "v1", "E3 no-DR", "stairs v11 fine-tune")
  │
  └──< run ──< stage (train | evaluate | stress | strict_test | gate | report)
        │        └──< stage_attempt (backend, external_id, status, started/ended, gpu_seconds, cost)
        ├── parent_run_id + parent_checkpoint (warm-start lineage)
        ├──< log_chunk (stage, seq, ts, stream, text)          -- capped, filtered
        ├──< metric_point (stage, step, key, value, wall_s)    -- from progress.csv / reporter
        ├──< artifact (kind: params|ckpt|video|csv|json|plot, uri, size, sha256, step)
        ├──< eval_result (checkpoint, suite, summary jsonb)    -- grid, stress, strict 96-run
        └── gate_decision (auto verdict + criteria results, reviewer, approved_at, comment)

compute_target (kind, name, config jsonb, encrypted_secret, quota, cost_per_gpu_hour, enabled)
dataset (for IL/teleop skills: demos, uri, robot, schema)     -- phase 3
audit_event (actor, action, entity, before/after, ip, ts)
webhook (url, events[], secret)                                -- integrations
```

Run lifecycle: `draft → pending_approval → queued → provisioning → running → evaluating → gated → (approved | rejected | failed | cancelled)`.

## 5. Skills: how existing and new skills plug in

A **skill** is one repo folder with a manifest plus an adapter. The web app never hard-codes a task.

```
skills/
  step_length/skill.yaml     # wraps g1pipe.train --task flat, scripts/eval_suite.py
  stairs/skill.yaml          # wraps g1pipe.train --task stairs, g1pipe.stairs_eval (strict 96-run)
  <new_skill>/skill.yaml
```

```yaml
id: g1-stairs
name: Stair climbing (vision-based)
robot: unitree-g1
category: locomotion            # locomotion | manipulation | workflow
method: rl-ppo + curriculum + height-scan   # links to the methods matrix (§6)
params:                          # pydantic model in adapter -> JSON Schema -> auto-generated form
  timesteps: {type: int, default: 200_000_000, min: 1_000_000}
  scan_model: {enum: [uniform, camera], default: camera}
  leg_action_scale: {type: float, default: 1.0}
  lr: {type: float, default: 1.0e-4}
  init_from: {type: checkpoint_ref, optional: true}   # warm start picker
pipeline:
  - {id: train,  kind: train,    requires: gpu, entrypoint: "g1pipe.train --task stairs"}
  - {id: strict, kind: evaluate, requires: cpu, entrypoint: "g1pipe.stairs_eval --layout test --starts 3",
     on: [best_checkpoints: 3, final]}
  - {id: gate,   kind: gate}
gate:                            # docs/pipeline.md §6, now machine-checked
  - {metric: strict.crossed_rate, op: ">=", value: 0.9}
  - {metric: strict.falls,        op: "<=", value: 5}
  - {metric: strict.certified_cm, op: ">=", value: 10}
```

**Adding a new skill** (for example pick-and-place or wiping a table) follows the same workflow the app guides you through:

1. **Propose** (UI): pick the task and category; the methods matrix suggests a method (RL for locomotion,
   teleop + imitation learning for manipulation, hybrid for workflows) and lists data needs.
2. **Implement** (repo): add the env and adapter under `skills/<id>/`, open a PR; CI runs `--smoke` on CPU.
3. **Register**: after merge, the skill appears as `draft` at that git SHA; an `ml_engineer` adds presets and
   gate criteria.
4. **Train, evaluate, and gate**: the same run pipeline as the existing skills.
5. **Release**: a `safety_reviewer` approves, and the release package is exported (§7).

Manipulation skills add a **dataset** step (upload or link teleop demos in LeRobot format) before training.
That's phase 3.

## 6. Methods comparison (the analysis half of the brief)

This is a page in the app, not a static slide. A matrix of methods (manual programming, teleop/LfD,
imitation learning, RL, sim-to-real with DR, human feedback, VLA/foundation models) scored against SKF's
requirements (data effort, training time, safety, repeatability, adaptability, compute, explainability,
deployment maturity), separately for locomotion, manipulation, and full workflows.

- The data lives in versioned YAML and is edited through a PR, so it can be reviewed.
- Each skill links to its method, and **measured evidence from real runs** (GPU hours, engineer hours, gate
  pass rate) is shown next to the theoretical score. For example, step length took 95 GPU-min and roughly
  one engineer-day, and stairs took 11 versions and about 9 GPU-h.
- Adjustable weights let SKF see which method ranks first for their priorities.

## 7. Connecting to SKF's existing systems

| SKF system | How we connect | Phase |
|---|---|---|
| Identity (likely Microsoft Entra ID) | Better Auth OIDC SSO; Entra group → role mapping; SCIM later | 2 |
| GPU compute | Pluggable `ComputeBackend`: Kaggle, AWS EC2 (existing scripts), **GCP Vertex AI custom jobs** (the brief's suggestion), SKF on-prem via SSH, Slurm, or a Kubernetes Job | 1–4 |
| Cloud accounts | Workload Identity Federation / IAM roles; no long-lived keys where avoidable; others encrypted at rest (envelope, KMS) | 2 |
| Storage | S3-compatible or GCS bucket in SKF's tenancy | 1 |
| Source code | Git: runs pinned to a commit SHA; trainer image tag = SHA | 1 |
| Experiment tracking | Optional MLflow export (runs, metrics, artifacts) | 3 |
| Notifications | Microsoft Teams / email webhooks: run finished, gate failed, approval needed | 2 |
| MES / PLM / ticketing | Versioned REST API + API keys + outgoing webhooks (`run.completed`, `release.approved`) | 3 |
| Robot deployment | **Export only**: a signed release package (policy weights + ONNX, obs/action spec, config, gate report, eval videos, SHA). SKF's unitree_sdk2 / ROS 2 toolchain consumes it. The web app never commands hardware. | 3 |

### Compute backends

```python
class ComputeBackend(Protocol):
    kind: str
    async def validate(self) -> HealthReport               # creds, quota, GPU availability
    async def estimate(self, spec: JobSpec) -> Estimate    # minutes, GPU-h, cost
    async def submit(self, spec: JobSpec) -> ExternalRef
    async def status(self, ref: ExternalRef) -> JobStatus
    async def logs(self, ref: ExternalRef, since: Cursor) -> LogBatch   # fallback when no phone-home
    async def collect(self, ref: ExternalRef, dest: ArtifactSink) -> list[Artifact]
    async def cancel(self, ref: ExternalRef) -> None
```

| Backend | Built from | Notes from our own runs |
|---|---|---|
| `local_cpu` | subprocess in the worker container | Smoke runs, cross-engine evals, videos (`MUJOCO_GL=egl`) |
| `kaggle` | `scripts/kaggle_job.py` logic, with `--init-kernel` for warm starts | Free T4 at about 12.5–36k steps/s; **30 GPU-h/week quota**, which gets tracked and shown; needs a phone-verified account |
| `aws_ec2` | `scripts/aws_box.sh` + `aws_bootstrap.sh` | L40S at about 100k steps/s once `warn_overflow=0`; capacity retries across instance types and AZs; idle auto-stop stays on |
| `gcp_vertex` | New: CustomJob with the pinned trainer image | Managed teardown, logs in Cloud Logging |
| `ssh` / `slurm` / `k8s` | New | For SKF on-prem GPUs |
| `colab_manual` | `scripts/make_colab.py` | No API: the app generates the notebook with a run token, and the user runs it; reporting works the same way |

**Log reporter** (tiny, stdlib-only, bundled into every job): it forwards stdout in batches, tails
`progress.csv` into metrics, sends heartbeats, and **drops known noise at the source** (the Warp solver
printf and `warnings.warn` lines already filtered in `kaggle_job.py`). It also caps log volume per run and
keeps the full raw log only as a compressed artifact. If there are no heartbeats for N minutes, the
orchestrator falls back to `backend.status()` and marks the run `stalled`.

## 8. UI (Next.js + shadcn)

| Route | Purpose | Roles |
|---|---|---|
| `/login` | Email + "Sign in with Microsoft" | all |
| `/` Dashboard | Active runs, GPU usage per target and quota left, gate pass rate, KPIs (GPU-h, cost, time-to-skill), awaiting approval | all |
| `/skills` | Catalog cards: status, method, best version, gate status | all |
| `/skills/[id]` | Spec, versions, presets, gate criteria, lineage graph (v11 ← v10@254M ← …), runs | all; edit: ml_engineer |
| `/runs/new` | 4-step wizard: skill → preset/params (form from JSON Schema) → compute target (live estimate, quota, cost) → review and launch | operator+ |
| `/runs` | Filterable table of runs | all |
| `/runs/[id]` | Pipeline stepper; tabs: **Logs** (live SSE, follow, filter by stage and level, search, download) · **Metrics** (reward, step_len_err, episode length) · **Checkpoints** (evaluate any checkpoint, warm start from it) · **Evaluation** (tracking grid heatmap, stress table, strict test) · **Videos** · **Gate** · **Config** | all; cancel/retry: owner, ml_engineer |
| `/compare` | Up to 4 runs side by side (for example E3: DR vs no-DR, stairs v9/v10/v11) | all |
| `/approvals` | Release gate queue: auto verdict, evidence, approve/reject with comment | safety_reviewer |
| `/methods` | Methods matrix with adjustable weights plus measured evidence | all |
| `/compute` | Targets: health check, quota, cost/h, credentials (write-only) | admin |
| `/admin/*` | Users and roles, API keys, webhooks, audit log | admin |

Screen mockups are shown in the conversation. The design is quiet and dense (a tool, not a marketing
site), in light and dark, with status colours used only for run and gate state.

## 9. Repository layout and code standards

```
apps/
  web/                    Next.js 16, pnpm, TypeScript strict, Biome
    src/app/(auth)/login
    src/app/(app)/{page,skills,runs,compare,approvals,methods,compute,admin}
    src/components/{ui (shadcn), runs, skills, charts, layout}
    src/lib/{auth.ts, auth-client.ts, permissions.ts, api/ (generated client), env.ts}
    src/proxy.ts
  api/                    FastAPI, uv, Python 3.12, ruff + pyright strict
    src/skf_api/
      main.py  settings.py
      core/        db, auth (JWKS verify), permissions, errors, logging, pagination
      modules/     skills/ runs/ compute/ artifacts/ gates/ methods/ audit/ ingest/ admin/
                   each: router.py  schemas.py  models.py  service.py  (routers thin, logic in service)
      backends/    base.py local.py kaggle.py aws_ec2.py gcp_vertex.py ssh.py colab_manual.py
      orchestrator/ state_machine.py  tasks.py (arq)  poller.py
    alembic/  tests/
  reporter/               stdlib-only log/metric reporter bundled into jobs
skills/                   skill manifests + adapters (step_length, stairs)
g1pipe/ jev_agent/ scripts/   existing code, unchanged apart from small hooks (e.g. eval_suite --out)
infra/
  docker/  api.Dockerfile  web.Dockerfile  trainer.Dockerfile (pinned jax 0.7.2 / brax 0.14.2 / mujoco 3.14)  evaluator.Dockerfile
  compose.yaml  compose.prod.yaml  Caddyfile
.github/workflows/  ci.yml (lint, typecheck, tests, OpenAPI drift, docker build)
```

Rules: no business logic in routers or React components; every external system sits behind an interface
with a fake for tests; settings come only from typed env (`pydantic-settings`, `@t3-oss/env-nextjs`); no
secrets in the browser or in logs; migrations are required for every schema change.

## 10. Production readiness

- **Security**: RBAC in the API, Better Auth CSRF/rate limiting, 2FA for admins, per-run scoped ingest
  tokens (HMAC, expire on run end), encrypted compute credentials, presigned URLs with short TTL, strict
  CSP, and only `/ingest` exposed publicly besides the web app.
- **Reliability**: an idempotent state machine (safe to re-run any transition), a poller that reconciles
  every non-terminal run, retries with backoff on provisioning (such as AWS capacity), cost guards (max
  GPU-h per run, idle auto-stop kept on AWS), and cancel that always tears down remote resources.
- **Observability**: structured JSON logs, OpenTelemetry traces (web → api → worker), `/healthz` and
  `/readyz`, Sentry.
- **Data**: nightly `pg_dump` to object storage, artifact bucket versioning, and a retention policy for raw logs.
- **Testing**: pytest (state machine, backends against fakes, permission matrix), Playwright e2e
  (login → launch smoke run on `local_cpu` → see logs → gate), and contract tests from OpenAPI.

## 11. Delivery phases

| Phase | Scope | Done when |
|---|---|---|
| **0 Foundations** | Monorepo, Compose (pg, redis, minio), Next.js + shadcn shell, Better Auth (email + roles + JWT), FastAPI skeleton with JWKS auth, CI | Staff can log in; the role matrix is enforced by API tests |
| **1 First pipeline end to end** | Skills registry (step length, stairs), run wizard, `local_cpu` + `kaggle` + `aws_ec2` backends, reporter, live logs + metrics, artifacts, eval stage, auto gate; import existing results (v1, nodr, stairs v1–v11) as historical runs | A smoke run launched from the UI streams logs and ends with a gate verdict; a Kaggle run does the same |
| **2 Team-ready** | Compare, lineage, approvals, KPI dashboard, Entra SSO, Teams notifications, GCP Vertex backend, budgets | A safety reviewer approves a release from the UI |
| **3 Extend** | Methods page with evidence, release package export, manipulation skill with datasets, MLflow export, public API + webhooks | A new skill is added via PR only, with no app code change |
| **4 Harden** | On-prem backends, pen-test fixes, backups drill, load test on log ingest | Go-live checklist is green |

For the hackathon demo, the slice to show is phase 0 plus a thin phase 1: log in, launch a stairs smoke run
on `local_cpu`, watch live logs, see the imported v11 results and gate verdict, and compare v9, v10, and v11.

## 12. Decisions needed from you

1. **SSO**: does SKF use Microsoft Entra ID? This plan assumes yes, with email/password as a fallback.
2. **GPU targets first**: Kaggle + AWS (both exist) in phase 1, with GCP in phase 2?
3. **Hosting**: where does the app run (SKF cloud tenancy, a VM, or our AWS for the pilot)? This decides
   whether `/ingest` can be reached from Kaggle and Colab.
4. **Scope for now**: implement phases 0–1 first and then review?
