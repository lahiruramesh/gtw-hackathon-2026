# SKF Skill Studio — implementation spec (phases 0–1 + team features)

This is the contract every part of the app is built against. `PLAN.md` explains *why*; this file says
*exactly what*. If code and this spec disagree, fix one of them in the same change.

Decisions from the product owner: **email + password only** (no SSO), compute locations **AWS EC2 and
Kaggle** (plus `local_cpu` for smoke runs and CPU evaluation), app **hosted on AWS**.

---

## 1. Repository layout (who owns what)

```
apps/api/                 FastAPI + worker (Python 3.12, uv)           -> owner: api-core
  src/skf_api/
    main.py               app factory, routers, CORS off (web proxies), error handlers
    settings.py           pydantic-settings, env prefix none (see §3)
    cli.py                `skf-api migrate|sync-skills|seed-targets|import-history|openapi` (targets are created via the API)
    core/                 db.py auth.py permissions.py errors.py logging.py pagination.py crypto.py
                          storage.py (S3/MinIO) events.py (Redis pub/sub) ingest_tokens.py
    modules/<name>/       router.py schemas.py models.py service.py   (routers thin; logic in service)
       skills/ runs/ compute/ artifacts/ gates/ audit/ ingest/ dashboard/ compare/
    skills_registry/      loader.py (reads skills/*/skill.yaml), manifest.py (pydantic), render.py (argv), summarize.py
    orchestrator/         state_machine.py tasks.py (arq WorkerSettings) reconciler.py executor.py collector.py
                          locking.py (row locks, stage lock) workdir.py (per-stage scratch)
                          worker: `uv run arq skf_api.orchestrator.tasks.WorkerSettings`
    history/              catalog.py importer.py effort_log.py (`skf-api import-history`)
    backends/             base.py (GIVEN, do not change the contract without updating this spec)
                          __init__.py (GIVEN)  local.py kaggle.py aws_ec2.py  -> owner: backends
  alembic/  alembic.ini
  tests/                  api-core tests; tests/backends/ -> owner: backends
apps/reporter/skf_reporter.py   stdlib-only in-job reporter                   -> owner: backends
skills/<skill>/skill.yaml, skills/<skill>/summarize.py                        -> owner: backends
apps/web/                 Next.js 16 + shadcn/ui + Better Auth                -> owner: web
infra/                    compose.yaml, compose.prod.yaml, docker/*.Dockerfile, Caddyfile, aws/  -> owner: infra
.github/workflows/ci.yml, Makefile, .env.example, docs/webapp/RUNBOOK.md      -> owner: infra
shared/permissions.json   roles -> permissions (GIVEN; single source for api and web)
g1pipe/ scripts/ jev_agent/  existing research pipeline. Only allowed change: scripts/eval_suite.py gets `--out` (backends)
```

## 2. Ports (local dev) and services

| Service | Container port | Host port (dev) |
|---|---|---|
| web (Next.js) | 3000 | **3100** (`pnpm dev -p 3100`) |
| api (uvicorn) | 8000 | **8000** |
| postgres 17 | 5432 | **55432** |
| redis 7 | 6379 | **56379** |
| minio (S3 API / console) | 9000 / 9001 | **59000 / 59001** |

Dev workflow: `make infra-up` starts postgres, redis, minio in Docker; api, worker and web run on the host
(`make api`, `make worker`, `make web`) for fast reloads. `docker compose --profile app up` runs everything
in containers. Docker CLI on this Mac: `/Applications/Docker.app/Contents/Resources/bin/docker`.

Postgres: one database `skf`, two schemas: `auth` (Better Auth tables, managed by Better Auth migrations)
and `app` (everything else, Alembic). `infra/postgres/init.sql` creates both schemas and a `skf_test` database.

## 3. Environment variables

API/worker (`apps/api/.env`, see `.env.example`):

| Var | Example | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://skf:skf@localhost:55432/skf` | |
| `REDIS_URL` | `redis://localhost:56379/0` | |
| `AUTH_JWKS_URL` | `http://localhost:3100/api/auth/jwks` | Better Auth JWKS |
| `AUTH_ISSUER` / `AUTH_AUDIENCE` | `skf-skill-studio` / `skf-api` | must match web |
| `SECRET_KEY` | 32+ random bytes, base64 | Fernet key derivation for target secrets + HMAC for ingest tokens |
| `S3_ENDPOINT_URL` | `http://localhost:59000` | empty on AWS (real S3) |
| `S3_BUCKET` / `S3_REGION` | `skf-artifacts` / `eu-north-1` | |
| `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | minio creds | empty on AWS (instance role) |
| `S3_PUBLIC_ENDPOINT_URL` | `http://localhost:59000` | host used in presigned URLs given to browsers |
| `PUBLIC_INGEST_URL` | `https://studio.example.com/ingest/v1` | empty = remote jobs don't phone home; worker polls logs |
| `INTERNAL_INGEST_URL` | `http://localhost:8000/ingest/v1` | used by local_cpu jobs (in compose: `http://api:8000/ingest/v1`) |
| `PIPELINE_REPO_DIR` | repo root | holds g1pipe/, scripts/, skills/, apps/reporter/ |
| `PIPELINE_PYTHON` | `<repo>/.venv/bin/python` | interpreter with `uv sync --extra train` deps |
| `WORK_DIR` | `/var/lib/skf/work` (dev: `<repo>/.skf-work`) | per-stage scratch |
| `LOG_MAX_LINES_PER_RUN` | `200000` | beyond this, lines are dropped and counted |
| `GIT_SHA` | set by the image build | commit recorded on skills/runs when the repo has no `.git` (containers) |
| `ENVIRONMENT` | `development` / `production` | |

Web (`apps/web/.env.local`): `DATABASE_URL` (plain `postgresql://…:55432/skf`), `BETTER_AUTH_SECRET`,
`BETTER_AUTH_URL` (`http://localhost:3100`), `API_INTERNAL_URL` (`http://localhost:8000`),
`AUTH_ISSUER`, `AUTH_AUDIENCE`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` (seed only).

## 4. Authentication and authorization

**Better Auth (web)** — `apps/web/src/lib/auth.ts`:
- `emailAndPassword: { enabled: true, disableSignUp: true, minPasswordLength: 12 }`. No public sign-up;
  admins create users. First admin: `pnpm seed:admin` (reads `ADMIN_EMAIL`/`ADMIN_PASSWORD`, idempotent).
- Database: `pg` `Pool` with `options: "-c search_path=auth"`. Migrations: `pnpm auth:migrate`
  (Better Auth CLI or `(await auth.$context).runMigrations()` — verify against the installed version).
- Plugins: `admin({ defaultRole: "viewer", adminRoles: ["admin"], roles })` where roles are the five from
  `shared/permissions.json` (Better Auth access control: `admin` gets admin statements, others user statements);
  `jwt({ jwks: { keyPairConfig: { alg: "EdDSA", crv: "Ed25519" } }, jwt: { issuer, audience, expirationTime: "15m",
  definePayload: ({ user }) => ({ email, name, role }) } })`; `nextCookies()` last.
- Rate limiting on (Better Auth built-in), session expiry 7 days, update age 1 day.

**API (FastAPI)** — `core/auth.py`:
- Every `/api/v1/*` route requires `Authorization: Bearer <jwt>`. Verify with PyJWT `PyJWKClient(AUTH_JWKS_URL)`
  (the key set is cached for 5 min; no per-key cache, so a key removed from the JWKS stops verifying within that time), algorithms `["EdDSA"]`, `issuer`, `audience`, `exp` required. Claims → `Principal(id=sub, email, name, role)`.
  Unknown role → 403. Banned users can't obtain tokens (Better Auth), tokens live 15 min.
- `core/permissions.py` loads `shared/permissions.json` (path from `PIPELINE_REPO_DIR` or relative to package) and
  exposes `require(permission)` dependency. Permission strings are exactly those in the JSON.
- `/me` returns the principal plus their permissions.

**Web → API**: server components and server actions call `apiFetch()` (`src/lib/api/server.ts`), which gets a JWT
via `auth.api.getToken({ headers: await headers() })` and calls `API_INTERNAL_URL`. Client components call the
same-origin proxy `app/api/backend/[...path]/route.ts`, which attaches the JWT and streams the upstream response
(required for SSE). The browser never sees the API origin or long-lived tokens.

**Permission matrix** (enforced by API, mirrored in UI):

| Action | Permission |
|---|---|
| read skills, runs, logs, metrics, artifacts, dashboard, compare | `skill:read` / `run:read` |
| sync skills from repo | `skill:write` |
| create run from preset (params == preset params, only `name`/`notes` differ) | `run:create_preset` |
| create run with custom params or warm start | `run:create_custom` |
| cancel/retry own run | `run:create_preset` and `created_by == me` |
| cancel/retry any run | `run:cancel_any` |
| approve/reject a `pending_approval` launch | `run:approve_launch` |
| evaluate a specific checkpoint | `run:evaluate` |
| review a release (approve/reject after gate pass) | `release:review` |
| list compute targets + usage | `compute:read` |
| create/update targets, set secrets, health check | `compute:write` |
| audit log | `audit:read` |
| user management (web, Better Auth admin API) | `user:manage` (admin role) |

Launch approval: a run goes to `pending_approval` instead of `queued` when the creator's role is `operator` and
`estimate.gpu_hours > target.max_unapproved_gpu_hours`. Everyone else is queued directly. The same rule applies to a
retry of a stage on the run's (GPU) target, judged by the retrier's role: the run goes back to `pending_approval`
(previous launch decision cleared, estimate refreshed) instead of `queued`.

## 5. Database schema (`app` schema, Alembic)

All ids are UUID v4 (server-generated) except `skills.id` (manifest slug). Timestamps are `timestamptz`, UTC.

- **skills**: `id text pk`, `name`, `summary`, `description`, `robot`, `category` (`locomotion|manipulation|workflow`),
  `method`, `status` (`draft|active|deprecated`), `manifest json` (json, not jsonb: jsonb reorders keys and the param order is the run form's field order), `manifest_sha text`, `git_sha text null`, `synced_at`.
- **compute_targets**: `id`, `name unique`, `kind` (`local_cpu|kaggle|aws_ec2`), `description`, `enabled bool`,
  `config jsonb`, `secret_enc bytea null` (Fernet), `gpu_label text` ("1x T4"), `steps_per_second float`,
  `overhead_minutes float`, `cost_per_gpu_hour float`, `weekly_quota_gpu_hours float null`,
  `max_unapproved_gpu_hours float`, `max_concurrent int`, `health jsonb null`, `health_checked_at null`, `created_at`, `updated_at`.
- **runs**: `id`, `name text unique` (slug `^[a-z0-9][a-z0-9-]{2,62}$`), `skill_id fk`, `preset_id text null`,
  `params jsonb`, `status` (see §6), `compute_target_id fk null`, `parent_run_id fk null`,
  `parent_checkpoint_id fk artifacts null`, `created_by_id`, `created_by_name`, `created_by_role`, `git_sha null`,
  `estimate jsonb null`, `notes text null`, `error text null`, `imported bool default false`,
  `cancel_requested bool default false`, `launch_decided_by_id/name null`, `launch_decided_at null`,
  `created_at`, `started_at null`, `finished_at null`. Indexes: `(skill_id, created_at desc)`, `(status)`.
- **stages**: `id`, `run_id fk cascade`, `key` (manifest stage id), `kind` (`train|evaluate|gate`), `title`,
  `position int`, `runs_on` (`target|local`), `status` (see §6), `compute_target_id fk null`, `attempt int default 1`,
  `external_ref jsonb null`, `log_cursor text null`, `progress float null`, `message text null`,
  `started_at null`, `finished_at null`, `last_heartbeat_at null`, `gpu_seconds float default 0`, `cost float default 0`,
  `noise_dropped bigint default 0`, `error text null`, `checkpoint_id fk artifacts null` (evaluate-a-checkpoint stages).
  Unique `(run_id, key, attempt)`.
- **log_lines**: `id bigserial pk`, `run_id`, `stage_id`, `ts`, `level` (`info|warn|error`), `text` (truncate 4000 chars).
  Index `(run_id, id)`. Level heuristic: `error` if line matches `Traceback|Error|ERROR|FAILED|Exception`, `warn` if `warn` (any case).
- **metric_points**: `id bigserial`, `run_id`, `stage_id`, `step bigint`, `key text`, `value double`, `wall_s double null`.
  Unique `(stage_id, key, step)`. Only keys matching the manifest's `metrics.keys` globs are stored (default `eval/*`
  without `_std` and without per-term `eval/episode_reward/*` unless listed).
- **artifacts**: `id`, `run_id`, `stage_id null`, `kind` (`params|checkpoint|video|image|csv|json|log|config|other`),
  `name` (relative path), `uri` (S3 key `runs/<run_id>/<stage_key>/<name>`), `size_bytes`, `sha256`, `content_type`,
  `step bigint null` (from `ckpt_<step>.pkl`), `created_at`. Unique `(run_id, uri)`. Every S3 object is tagged
  `skf-kind=<kind>` (the bucket lifecycle moves and expires `log` objects by that tag).
- **evaluations**: `id`, `run_id`, `stage_id`, `checkpoint_id fk artifacts null`, `suite text`, `summary jsonb`, `created_at`.
- **gate_decisions**: `run_id pk fk`, `verdict` (`pass|fail`), `criteria jsonb`, `evaluated_at`,
  `review_status` (`pending|approved|rejected|not_required`), `reviewer_id/name null`, `comment null`, `reviewed_at null`.
- **audit_events**: `id bigserial`, `ts`, `actor_id`, `actor_name`, `actor_role`, `action` (e.g. `run.create`,
  `run.cancel`, `run.launch_approve`, `release.approve`, `target.update`, `target.secret_set`, `user.create`),
  `entity_type`, `entity_id`, `detail jsonb`, `ip text null`. Never store secrets in `detail`.

## 6. Run and stage state machine

Run status: `pending_approval → queued → running → (awaiting_review → approved | rejected) | gate_failed | failed | cancelled`.
A launch rejected by an approver → `cancelled` with `error = "launch rejected: <comment>"`.

Stage status: `pending → queued → provisioning → running → collecting → succeeded | failed | cancelled | skipped`.

Orchestrator (arq worker, `orchestrator/tasks.py`):
- `advance_run(run_id)`: lock the run row (`SELECT … FOR UPDATE`). If `cancel_requested` → cancel. If any stage
  failed → run `failed`. Otherwise start the first `pending` stage: `gate` stages are evaluated inline (§8);
  others move to `queued` and `execute_stage` is enqueued. When all stages succeeded and gate evaluated → final status.
- `execute_stage(stage_id)`: respect `target.max_concurrent` (stay `queued`, retry in 30 s). Build `JobSpec`
  (§7), download inputs from S3, call `backend.submit()`, persist `external_ref`, set `provisioning`/`running`.
  The call holds the Redis stage lock (60 s TTL, renewed while held, so a dead worker frees it within a minute). If
  the stage was finished elsewhere during the submit, the new job is cancelled and the target released; if the run
  was cancelled meanwhile, the job is recorded and `cancel_run` enqueued. A failed submit releases the target (it may
  have booted the box).
  For `local_cpu` the backend streams stdout itself into the log sink (see backends) and returns quickly; the
  process is supervised by the backend inside the worker process.
- `reconcile()` (arq cron, every 20 s): for every stage in `provisioning|running`: `backend.status()`; if no ingest URL
  configured, `fetch_logs()` and append; update `gpu_seconds`, `cost = gpu_seconds/3600*target.cost_per_gpu_hour`,
  progress. On success → `collecting`: `collect()` into `WORK_DIR/<run>/<stage>/out`, upload every file as an
  artifact (checkpoints as `checkpoint` with `step`), parse `progress.csv` into metrics if the reporter didn't,
  run the skill summarizer for `evaluate` stages → `evaluations` row, mark `succeeded`, enqueue `advance_run`, delete
  the stage's `WORK_DIR` scratch (also for failed/cancelled stages), then `backend.release()` if no other active stage
  on that target. Stages with no heartbeat for 10 min on a phone-home job → poll status; local stages whose
  supervising worker died → `failed` ("worker restarted"). `cancel_run` does not take the stage lock, so after the
  (slow) backend calls every transition re-locks the stage row (`populate_existing`) and only proceeds if the stage
  is still in the expected status and the run is not being cancelled. Collected files are regular files only:
  symlinks in a job's outputs are dropped, never followed. Each tick also enqueues `advance_run` for `queued`/`running`
  runs with no active main stage (their `advance_run` was lost), and `cancel_run` for runs whose cancel is unfinished.
- `cancel_run(run_id)`: `backend.cancel()` for active stages (idempotent), then stages → `cancelled`, run → `cancelled`.
  A stage is marked cancelled only when nothing can still run for it: its cancel succeeded or failed non-retryably, or
  it never reached a backend. A stage whose submit is in flight is left to `execute_stage`; after a retryable cancel
  error the stage stays active (message "Stopping the job failed…") and the reconciler retries with backoff
  (30 s doubling to 10 min, 10 attempts, then the stage is cancelled with the error recorded). The API cancels a run
  directly only while no stage has started; otherwise it enqueues `cancel_run`.
- Every transition: update row, publish a `stage`/`run` event (§9.3), write audit for user-initiated ones.
- Retry: `POST /runs/{id}/retry` creates a new attempt of the failed stage (`attempt+1`) and resets later stages to `pending`.
  Run views and the state machine use the latest attempt per stage key.
- Checkpoint evaluations (`POST /runs/{id}/evaluate`) are *side stages* (`checkpoint_id` set): queued directly,
  never change the run's status and never re-run the gate; their evaluations are stored under their own key.

## 7. Skill manifests (`skills/<dir>/skill.yaml`)

Loaded at API startup and via `POST /skills/sync`. Validated by `skills_registry/manifest.py` (pydantic). Shape:

```yaml
id: g1-step-length                    # slug, primary key
name: Step-length adjustment
summary: Walk at a commanded speed and step length.
description: |                        # markdown
  ...
robot: unitree-g1
category: locomotion
method: rl-ppo                        # key into the methods matrix
status: active
params:                               # JSON-Schema-like; drives the UI form and validation
  timesteps: {type: integer, title: Training steps, default: 200000000, minimum: 20000, maximum: 2000000000}
  smoke:     {type: boolean, title: Smoke test (tiny CPU run), default: false}
  no_dr:     {type: boolean, title: Disable domain randomisation, default: false}
  lr:        {type: number, title: Learning rate, default: null, nullable: true}
  seed:      {type: integer, title: Seed, default: 0}
  init_from: {type: checkpoint, title: Warm start from, nullable: true, default: null}
presets:
  - {id: smoke, name: Smoke test, description: 90 s CPU check of the whole pipeline, params: {smoke: true, timesteps: 20000}}
  - {id: v1, name: v1 (with randomisation), params: {timesteps: 200000000}}
  - {id: e3-no-dr, name: E3 without randomisation, params: {timesteps: 200000000, no_dr: true}}
pipeline:
  - id: train
    kind: train
    title: Train (Brax PPO)
    runs_on: target                   # the run's compute target
    argv: [python, -m, g1pipe.train, --out, "{out_dir}", --timesteps, "{timesteps}", --seed, "{seed}"]
    flags: {smoke: --smoke, no_dr: --no-dr, lr: --lr, init_from: --init-from}
    progress_csv: progress.csv
  - id: evaluate
    kind: evaluate
    title: Cross-engine evaluation and stress tests
    runs_on: local                    # local_cpu on the worker host
    estimate_minutes: 15
    argv: [python, scripts/eval_suite.py, "{input_dir}/train/params.pkl", --out, "{out_dir}", --quick]
    inputs: [train/params.pkl, train/config.json]
    suite: e2e-stress
  - id: gate
    kind: gate
    title: Release gate
gate:
  - {metric: evaluate.grid_fall_rate, op: "==", value: 0, label: No falls on the command grid}
  - {metric: evaluate.grid_step_abs_err_cm, op: "<", value: 3, label: Step-length error < 3 cm (unseen engine)}
  - {metric: evaluate.stress.push_0.5mps.fall_rate, op: "==", value: 0, label: Survives 0.5 m/s pushes}
  - {metric: evaluate.stress.latency_20ms.fall_rate, op: "==", value: 0, label: Survives 20 ms latency}
headline:                             # shown in run tables and compare
  - {label: Step error, metric: evaluate.grid_step_abs_err_cm, unit: cm, digits: 1}
  - {label: Grid falls, metric: evaluate.grid_fall_rate, unit: "%", scale: 100, digits: 0}
metrics:
  keys: ["eval/episode_reward", "eval/episode_step_len_err", "eval/avg_episode_length", "eval/episode_crossed", "eval/sps"]
  primary: eval/episode_reward
```

Rendering (`skills_registry/render.py`): `argv` entries are literals or `{param}` placeholders (resolved from
params with defaults applied); `{out_dir}` and `{input_dir}` are left for the backend. `flags`: boolean true →
flag appended; non-null value → flag + value; `checkpoint` params resolve to `{input_dir}/parent/<file>` and add
an `InputFile` from the chosen artifact. Values are stringified; no shell is ever used (argv lists only).
Summaries: `skills/<dir>/summarize.py` exposes `summarize(out_dir: Path) -> dict` (pure stdlib/json/csv), loaded
by file path. The evaluation summary is stored under the stage key, so gate metric `evaluate.stress.push_0.5mps.fall_rate`
means `evaluations[stage=evaluate].summary["stress"]["push_0.5mps"]["fall_rate"]` (dot path; list-of-dicts with a
`case` key is converted to a dict keyed by `case` by the summarizer).

Stairs skill (`skills/stairs/skill.yaml`): `id: g1-stairs`, train argv `--task stairs` with params `timesteps`,
`smoke`, `scan_model` (enum uniform|camera), `leg_action_scale`, `lr`, `seed`, `init_from`; presets `smoke`,
`v11-finetune` (lr 1e-4, leg_action_scale 1.0, scan camera, 300M); evaluate stage runs
`python -m g1pipe.stairs_eval {input_dir}/train/params.pkl --starts 3 --out {out_dir}/strict.json`
(estimate 25 min); summary: `n`, `crossed`, `fell`, `crossed_rate`, `fall_rate`, `certified_cm`
(same rule as `g1pipe.stairs_eval.certified_height`: GATE_TILT_DEG 25, GATE_PELVIS_M 0.55), `by_height` list;
gate: `crossed_rate >= 0.9`, `fell <= 5`, `certified_cm >= 10`. Headline: Crossed (%), Falls, Certified height (cm).

## 8. Simulation and release gates

Each manifest criterion has a `level`: `simulation` (the policy works in simulation under nominal conditions) or
`release` (default; the bar before gantry and floor trials, docs/pipeline.md §6). The simulation gate passes when its
criteria pass; the **release gate needs every criterion**, simulation ones included, and is the run's verdict.
`GateDecision.levels` gives both verdicts with passed/total counts; `RunSummary.simulation_verdict` sits next to
`gate_verdict` (the release verdict). Decisions stored before levels existed count as release only.
`skf-api regate [RUN...]` re-evaluates unreviewed decisions (and the gate stage's title and message) after gate
criteria change; approved or rejected decisions are left alone.

`gates/service.py::evaluate_gate(run)`: for each criterion resolve the metric (latest evaluation of that stage key),
compare with `op` (`< <= > >= == !=`, floats compared with 1e-9 tolerance for `==`). Missing metric → `passed=false,
actual=null`. Verdict pass → run `awaiting_review`, `review_status=pending`. Fail → run `gate_failed`,
`review_status=not_required`. `POST /runs/{id}/review {decision: "approve"|"reject", comment}` (release:review,
comment required on reject, reviewer can't be the run creator) → `approved`/`rejected`.

## 9. HTTP API

Base `/api/v1`. JSON, snake_case. Errors: `{"error": {"code": "not_found", "message": "Run not found", "details": {}}}`
with proper status (400 validation → 422 with pydantic details in `details`, 401, 403 `forbidden`, 404, 409 `conflict`).
Lists: `{"items": [...], "next_cursor": "..."|null}` with `?limit=` (default 50, max 200) and `?cursor=`.

### 9.1 Resources (response shapes)

```
Me            {id, email, name, role, permissions: string[]}
UserRef       {id, name}
RunRef        {id, name, status}
SkillSummary  {id, name, summary, robot, category, method, status, git_sha, run_count, last_run_at|null,
               best_run: RunRef|null}            # latest approved, else latest gate pass, else latest succeeded
SkillDetail   SkillSummary + {description, params_schema: JSONSchema (checkpoint params carry `"x-kind": "checkpoint"`), presets: Preset[], pipeline: PipelineStageDef[],
               gate: GateCriterion[], headline: Headline[], metrics: {keys: string[], primary: string}}
Preset        {id, name, description|null, params: object}
PipelineStageDef {id, kind, title, runs_on}
GateCriterion {metric, op: "<"|"<="|">"|">="|"=="|"!=", value, label, level: "simulation"|"release"}
Headline      {label, metric, unit|null, scale|null, digits|null}
ComputeTarget {id, name, kind, description|null, enabled, config: object, has_secret: bool, gpu_label|null,
               steps_per_second, overhead_minutes, cost_per_gpu_hour, weekly_quota_gpu_hours|null,
               max_unapproved_gpu_hours, max_concurrent,
               health: {status: "ok"|"degraded"|"down"|"unknown", message, checked_at|null},
               usage: {gpu_hours_7d, cost_7d, active_stages, quota_left_hours|null, quota_source: "ledger"|"provider"|null},
               # quota_left_hours = min(weekly_quota_gpu_hours - gpu_hours_7d [ledger],
               #   health.details.gpu_remaining_hours reported at the last check [provider, e.g. Kaggle])
               created_at, updated_at}
Estimate      {train_minutes, total_minutes, gpu_hours, cost, needs_approval: bool, reasons: string[], warnings: string[],
               blockers: string[]}               # blockers: why POST /runs would refuse (disabled, no credentials, local training)
Stage         {id, key, kind, title, position, runs_on, status, compute_target_id|null, attempt, progress|null,
               message|null, started_at|null, finished_at|null, last_heartbeat_at|null, gpu_seconds, cost,
               noise_dropped, error|null, external_url|null, checkpoint: Artifact|null}
RunSummary    {id, name, skill_id, skill_name, preset_id|null, status, compute_target: {id,name,kind}|null,
               created_by: UserRef, created_at, started_at|null, finished_at|null,
               current_stage: {key, title, status, progress|null}|null, gate_verdict: "pass"|"fail"|null,
               parent: {run: RunRef, checkpoint: Artifact|null}|null,   # warm-start lineage, also in lists
               imported, gpu_hours, cost, headline: {label, value: number|null, unit|null}[]}
RunDetail     RunSummary + {params, git_sha|null, notes|null, error|null, estimate: Estimate|null,
               children: RunRef[], stages: Stage[],
               gate: GateDecision|null, launch_decided_by: UserRef|null, launch_decided_at|null,
               permissions: {can_cancel, can_retry, can_approve_launch, can_review, can_evaluate}}
LogLine       {id, stage_id, ts, level: "info"|"warn"|"error", text}
Artifact      {id, stage_id|null, stage_key|null, kind, name, size_bytes, content_type, step|null, created_at}
Evaluation    {id, stage_id, stage_key, checkpoint: Artifact|null, suite, summary: object, created_at}
GateDecision  {verdict, criteria: {metric, label, op, value, actual: number|null, passed: bool}[], evaluated_at,
               review_status, reviewer: UserRef|null, comment|null, reviewed_at|null}
AuditEvent    {id, ts, actor: {id, name, role}, action, entity_type, entity_id|null, detail: object}
```

### 9.2 Endpoints

| Method | Path | Permission | Body / query → response |
|---|---|---|---|
| GET | `/healthz` `/readyz` (at the root, not under `/api/v1`) | public | `{status, checks}`; readiness checks db + redis (503 if either fails) |
| GET | `/me` | any | `Me` |
| GET | `/skills` | skill:read | `Page<SkillSummary>` |
| GET | `/skills/{id}` | skill:read | `SkillDetail` |
| POST | `/skills/sync` | skill:write | reload manifests → `{synced: string[], errors: {file, message}[]}` |
| GET | `/compute-targets` | compute:read | `ComputeTarget[]` (secrets never returned) |
| GET | `/compute-targets/schema/{kind}` | compute:write | `{config: JSONSchema, secret: JSONSchema}` |
| POST | `/compute-targets` | compute:write | create `{name, kind, description?, enabled, config, gpu_label?, steps_per_second, overhead_minutes, cost_per_gpu_hour, weekly_quota_gpu_hours?, max_unapproved_gpu_hours, max_concurrent}` → `ComputeTarget` |
| PATCH | `/compute-targets/{id}` | compute:write | partial of the above → `ComputeTarget` |
| PUT | `/compute-targets/{id}/secret` | compute:write | `{secret: object}` → 204 (write-only; validated against backend secret model) |
| POST | `/compute-targets/{id}/check` | compute:write | runs `backend.validate()` → `ComputeTarget` |
| POST | `/runs/estimate` | run:create_preset | `{skill_id, preset_id?, params, compute_target_id}` → `Estimate` |
| GET | `/runs` | run:read | `?skill_id&status&created_by=me&q&limit&cursor` → `Page<RunSummary>` |
| POST | `/runs` | run:create_preset (+custom rules §4) | `{skill_id, preset_id?, params, compute_target_id, name?, notes?, parent_run_id?, parent_checkpoint_id?}` → 201 `RunDetail`. Name default `<skill>-<preset or custom>-<n>`. 409 on duplicate name. Target must be enabled, have a secret if its kind needs one, and not be `local_cpu` for non-smoke train unless `ENVIRONMENT=development` (exactly the estimate's `blockers`; 422 with the first one). |
| GET | `/runs/{id}` | run:read | `RunDetail` |
| POST | `/runs/{id}/cancel` | own or run:cancel_any | → `RunDetail` |
| POST | `/runs/{id}/retry` | own or run:cancel_any | only when `failed` → `RunDetail` (`queued`, or `pending_approval` per §4) |
| POST | `/runs/{id}/launch-decision` | run:approve_launch | `{decision: "approve"|"reject", comment?}` → `RunDetail` (not own run) |
| POST | `/runs/{id}/review` | release:review | `{decision, comment?}` → `RunDetail` |
| POST | `/runs/{id}/evaluate` | run:evaluate | `{checkpoint_id, stage_key?}` → adds an `evaluate` stage bound to that checkpoint (key `<stage_key>-ckpt-<step>`, e.g. `evaluate-ckpt-254279680`; `…-ckpt-final` for `params.pkl`), runs it; → `RunDetail` |
| GET | `/runs/{id}/logs` | run:read | `?stage_id&after_id&limit(≤2000)&level&q` → `{items: LogLine[], next_after_id|null}` |
| GET | `/runs/{id}/logs.txt` | run:read | `text/plain` attachment of all lines |
| GET | `/runs/{id}/metrics` | run:read | `?keys=a,b&stage_id` → `{keys: string[], series: {key, points: [step, value][]}[]}` (downsampled to ≤1000 pts) |
| GET | `/runs/{id}/artifacts` | run:read | `Artifact[]` |
| GET | `/artifacts/{id}/url` | run:read | `{url, expires_at}` presigned GET (15 min, `S3_PUBLIC_ENDPOINT_URL` host) |
| GET | `/runs/{id}/evaluations` | run:read | `Evaluation[]` |
| GET | `/runs/{id}/events` | run:read | **SSE**, see 9.3 |
| GET | `/approvals` | release:review or run:approve_launch | `{launches: RunSummary[], releases: RunSummary[]}` filtered by the caller's permissions |
| GET | `/compare` | run:read | `?run_ids=a,b,c,d` (2–4) → `{runs: RunDetail[], headline_rows: {label, unit, values: (number|null)[]}[], gate_rows: {label, values: (bool|null)[]}[], metric_keys: string[]}` (gate value `null`: no gate, or the metric was never measured) |
| GET | `/dashboard` | run:read | `{active_runs, awaiting_launch_approval, awaiting_review, gpu_hours_7d, cost_7d, gate_pass_rate_30d|null, targets: {id, name, kind, gpu_label, active_stages, quota_left_hours|null, health}[], recent_runs: RunSummary[], skills: {id, name, runs, gpu_hours_total, best_run: RunRef|null, latest_verdict|null}[]}` |
| GET | `/audit-events` | audit:read | `?actor_id&action&entity_type&cursor` → `Page<AuditEvent>` |
| POST | `/audit-events` | user:manage | `{action, entity_type, entity_id?, detail}` → 201 (web records user-management actions) |

Ingest (public; mounted at `/ingest/v1`, not under `/api/v1`; token auth only):

| Method | Path | Body |
|---|---|---|
| POST | `/ingest/v1/stages/{stage_id}/logs` | `{lines: [{ts?: ISO8601, text, stream?: "stdout"|"stderr"}], noise_dropped?: int}` (≤1000 lines, ≤1 MB) |
| POST | `/ingest/v1/stages/{stage_id}/metrics` | `{points: [{step, wall_s?, values: {key: number}}]}` |
| POST | `/ingest/v1/stages/{stage_id}/heartbeat` | `{progress?: 0..1, message?}` |

All three answer `{accepted: int, dropped: int}` (lines/values stored, lines dropped by the per-run cap); 413 over 1 MB,
422 over 1000 lines, 429 over the rate limit. The first ingest call of a `provisioning` stage moves it to `running`.

Ingest token (`core/ingest_tokens.py`): `base64url(stage_id) + "." + exp_unix + "." + hex(hmac_sha256(SECRET_KEY_derived, f"{stage_id}.{exp}"))`,
exp = submit time + stage timeout + 1 h. Valid only if signature matches, not expired, path stage_id matches and the stage
is not terminal. 401 otherwise. Rate limit per stage: 20 req/s (Redis counter). All three endpoints update `last_heartbeat_at`.

### 9.3 Server-sent events — `GET /runs/{id}/events?after_log_id=<n>`

`Content-Type: text/event-stream`; first replays log lines with `id > after_log_id` (max 2000, then the client pages
via `/logs`), then streams live events from Redis channel `run:<run_id>`. The stream opens with a `: connected` comment
(uvicorn sends headers only with the first body chunk), then a comment heartbeat `: ping` every 15 s.
Each event: `event: <type>\nid: <log id for logs>\ndata: <json>\n\n`. Types:
`log` → `LogLine`; `metric` → `{stage_id, step, values: {key: number}}`; `stage` → `Stage`; `run` → `RunSummary`.
Publishers: log sink (batched: one Redis message may carry many lines → emit one `log` event per line), metric
writer, orchestrator transitions.

## 10. Backends (owner: backends) — contract in `backends/base.py`

Common: `Config` and `Secret` pydantic models per backend (used by `config_schema()`), constructor `(ctx: TargetContext)`.
`TargetContext` carries the target row (config, decrypted secret) plus the worker's `work_dir`, `pipeline_repo_dir` and
`pipeline_python`: backends read settings only through it, never from `os.environ`.

- **local_cpu** (`LocalCpuConfig {max_concurrent_hint: int = 2, python: str|None, work_subdir: "local"}`, no secret): runs
  argv with `config.python`, else `TargetContext.pipeline_python` (the `PIPELINE_PYTHON` setting), else the worker's interpreter, `cwd=pipeline_repo_dir`, env
  `PYTHONPATH=<repo>`, `MUJOCO_GL=egl` on Linux, plus only an allowlist of worker variables (PATH, HOME, locale, `XLA_*`,
  `JAX_*`, `MUJOCO_*`, `OMP_*`: the worker's own secrets never reach the job); replaces `{out_dir}`/`{input_dir}` with
  `<ctx.work_dir>/<work_subdir>/<run_id>/<stage_key>-<stage_id[:8]>/{out,in}` (inputs are copied in). The job is wrapped
  in the reporter exactly like on AWS (the reporter itself runs on the worker's interpreter):
  `python3 apps/reporter/skf_reporter.py --log <work>/job.log --exit-file <work>/exit_code --progress <out>/progress.csv -- <pipeline_python> ...`.
  Local jobs always phone home to `INTERNAL_INGEST_URL` (the orchestrator sets `JobSpec.ingest` for local stages even when
  `PUBLIC_INGEST_URL` is empty), so their logs are live. `fetch_logs()` (tail `job.log` by byte offset) stays as the fallback;
  `status()` checks the pid (and its command line, so a reused pid is not mistaken for the job) and exit file. The process
  is started detached (`start_new_session=True`) with its pid in the ref; `cancel()` marks the stage cancelled and kills the
  process group (SIGTERM, then SIGKILL after a grace period). `collect()` copies `out/` plus `job.log`. `release()` deletes
  the job dirs of finished jobs (exit file, or cancelled and no longer alive).
- **kaggle** (`KaggleConfig {username, accelerator: "NvidiaTeslaT4", kernel_prefix: "skf", enable_internet: true}`,
  `KaggleSecret {key}`): builds a single-file private script kernel like `scripts/kaggle_job.py` (import `PINS` and
  `BUNDLE` from that file via importlib so there is one source of truth; `NOISE` is read from its kernel template and passed
  to the reporter as `--noise`), embedding `apps/reporter/skf_reporter.py`
  and the rendered argv; `{out_dir}` = `/kaggle/working/run`; warm start: `InputFile.source_ref` with a Kaggle ref →
  `kernel_sources` + locate `params.pkl` under `/kaggle/input` (as the existing template does); inputs without a Kaggle
  source → `BackendError("warm start on Kaggle needs a parent trained on Kaggle")`. Uses the `kaggle` CLI with
  `KAGGLE_USERNAME`/`KAGGLE_KEY` env (never written to disk outside a temp `KAGGLE_CONFIG_DIR` with 0600).
  Kernel slug `<kernel_prefix>-<run_name>-<stage_key>` (≤41 chars) `-<stage_id[:8]>`; submit checks `kernels status` first
  and never pushes again over an existing kernel; `kernels push --accelerator <acc> --timeout <stage timeout>`.
  status via `kaggle kernels status`; collect via `kaggle kernels output -p dest` (then files from `run/` are the outputs,
  and the reporter's `job.log` — or, if the kernel died before it started, Kaggle's `<slug>.log` — becomes `job.log`);
  fetch_logs: nothing until complete, then `job.log` once. gpu_seconds = wall time while running. Only `train` stages.
  cancel(): the public Kaggle API cannot stop a session, so the kernel's reporter runs with `--stop-when-revoked` (stops the
  job once ingest keeps rejecting heartbeats because the stage is terminal) and the push timeout is a hard limit.
  validate(): `kaggle kernels list --mine --page-size 1`, plus `kaggle quota --csv` for GPU hours left (degraded < 1 h;
  `details.gpu_remaining_hours` feeds the target's `quota_left_hours`).
- **aws_ec2** (`AwsEc2Config {region, instance_name, fallback_instance_names: [], instance_type, ami_id, subnet_id, security_group_id, key_name,
  ssh_user: "ubuntu", remote_repo_dir: "~/gtw", bootstrap_command: "MJX_ONLY=1 bash ~/gtw/scripts/aws_bootstrap.sh",
  volume_gb: 250, create_if_missing: false, stop_when_idle: true, aws_profile: str|None}`, `AwsEc2Secret {ssh_private_key,
  aws_access_key_id?, aws_secret_access_key?, aws_session_token?}` — empty keys = default credential chain / instance role):
  mirrors `scripts/aws_box.sh`: find instance by Name tag (primary, then `fallback_instance_names`, e.g. the same box in
  other availability zones: a running candidate is reused, otherwise the first one AWS has capacity to start), start if stopped (retry capacity errors as
  `BackendError(retryable=True)`), wait for SSH, `rsync` the pipeline repo (g1pipe, jev_agent, scripts, skills,
  apps/reporter, pyproject.toml, uv.lock; exclude runs/.venv/third_party) at the current commit, upload inputs via `scp`,
  upload `run.sh`/`job.sh` and start `tmux new-session -d -s skf-<stage8> bash ~/skf/<stage_id>/run.sh`: `run.sh` runs
  `python3 ~/gtw/apps/reporter/skf_reporter.py --log ~/skf/<stage_id>/job.log --exit-file ~/skf/<stage_id>/exit_code --timeout <s> --progress <out>/progress.csv -- bash job.sh`,
  and `job.sh` runs the bootstrap once per box (marker `~/skf/.bootstrapped`, under `flock`, so its output is in the stage log)
  then `uv run --no-sync python ...` (no sync: keep the bootstrapped CUDA wheels) with `out_dir=~/skf/<stage_id>/out`.
  status: `cancelled` marker / exit file / `tmux has-session` (a finished stage gets a `reported` marker; the exit code is kept
  in the ref, so a later stop does not lose it); fetch_logs: `tail -c +<offset>` (≤1 MB); collect: rsync `out/` and `job.log`
  down with `--no-links` (starting the box if it was stopped); cancel: `cancelled` marker + `tmux kill-session`; release:
  delete the dirs of `reported` stages (collected by then), then stop the instance if `stop_when_idle`, no `skf-*` tmux
  session and no finished-but-unreported stage. gpu_seconds = instance running time
  attributable to the stage. All boto3 calls via `asyncio.to_thread`. SSH key written to a 0600 temp file, deleted after use;
  `StrictHostKeyChecking=accept-new` with a per-target known_hosts, keyed by `HostKeyAlias=<instance id>` (the IP changes).
  validate(): describe the instance (state, type) — never starts it.

**Reporter** (`apps/reporter/skf_reporter.py`, stdlib only, Python ≥3.8): `skf_reporter.py [--log FILE] [--exit-file FILE]
[--progress CSV] [--noise SUBSTR]... [--timeout SECONDS] [--stop-when-revoked] -- <cmd...>`. Runs cmd, merges stdout/stderr,
drops noise lines (defaults: the `NOISE` tuple in `scripts/kaggle_job.py` — keep the two lists identical), writes kept lines
to `--log` and stdout, counts dropped lines; if `SKF_INGEST_URL`/`SKF_INGEST_TOKEN`/`SKF_STAGE_ID` are set, POSTs batches
(every 2 s or 500 lines; `noise_dropped` = lines dropped since the previous batch) to the ingest API with retries (exponential
backoff, full jitter; 4xx other than 408/429 are not retried), tails the progress CSV into `/metrics` (numeric columns only,
`step` and `wall_s` taken out), and heartbeats every 30 s (`{message}`: the latest step, e.g. `"step 2,000"`, or empty); ingest failures never affect the child. SIGTERM,
SIGINT and SIGHUP are forwarded to cmd (SIGKILL after 20 s). `--timeout` stops cmd and exits 124; `--stop-when-revoked` stops
it (exit 143) after 3 heartbeats in a row get 401/403/404/410. Exit code = child's (128+N if killed by signal N); also
written to `--exit-file`.

## 11. Web app (owner: web)

Next.js 16 App Router, TypeScript strict, `src/` dir, Tailwind v4, shadcn/ui (new-york, neutral base, CSS variables),
`next-themes` (light/dark/system), `lucide-react` icons, `recharts` via shadcn `chart`, `react-hook-form` + `zod`,
`sonner` toasts, `@tanstack/react-query` only where client polling/mutation state helps. pnpm. Biome or ESLint+Prettier.
`src/proxy.ts` (Next 16 replacement for middleware) redirects unauthenticated users to `/login` (cookie check only;
real checks happen server-side per page with `auth.api.getSession`).

Layout: left sidebar (collapsible, shadcn `sidebar`), top bar with breadcrumb, role badge, user menu (theme, sign out).
Nav items hidden when the role lacks the permission. Design: quiet, dense, tool-like; status colours only for
run/stage/gate state; tabular numbers; empty states with a clear next action; skeleton loading; error boundaries.

| Route | Content |
|---|---|
| `/login` | Email + password, error states, no sign-up link ("No account? Ask an admin to create one.") |
| `/` | Metric cards (active runs, GPU-h 7d, cost 7d, awaiting approval/review), compute target cards with quota, recent runs table, per-skill summary |
| `/skills` | Skill cards (category, method, status, best run, headline metrics) |
| `/skills/[id]` | Tabs: Overview (description markdown, pipeline diagram as stepper, gate criteria), Presets, Runs (table), Lineage (tree of runs by `parent_run_id`) ; "New run" button |
| `/runs` | Table with filters (skill, status, mine), headline columns, status badges, pagination |
| `/runs/new` | Wizard (`?skill=&preset=&parent=`): 1 Skill → 2 Preset or custom params (form generated from `params_schema`; custom only with `run:create_custom`; checkpoint picker lists the parent's checkpoints) → 3 Compute target cards with live `Estimate` (quota, cost, approval warning) → 4 Review (name, notes) and Launch |
| `/runs/[id]` | Header (name, status, target, git sha, actions: cancel, retry, approve launch, review). Pipeline stepper with per-stage status/progress/duration. Tabs: **Logs** (SSE live, follow toggle, stage filter, level filter, search highlight, virtualised list, download, "N noise lines dropped"), **Metrics** (line charts per selected key, primary first), **Checkpoints** (artifact list of `checkpoint`/`params`, "Evaluate" and "Warm start new run" actions), **Evaluation** (summary tables: grid heatmap for step-length `grid` if present, stress table, stairs by-height table; generic JSON viewer fallback), **Artifacts** (videos play inline via presigned URL, images, downloads), **Gate** (criteria table with actual vs threshold, verdict, review form for reviewers), **Config** (params, estimate, parent) |
| `/compare` | Pick 2–4 runs; headline table, gate table, overlaid metric chart |
| `/approvals` | Two lists: launches awaiting approval, releases awaiting review; approve/reject dialogs with comment |
| `/methods` | Learning-methods matrix (manual programming, teleop/LfD, imitation learning, RL, sim-to-real + DR, human feedback, VLA/foundation models) × requirements (data effort, training time, safety, repeatability, adaptability, compute, explainability, deployment maturity), scored 1–5 per domain (locomotion / manipulation / workflow), adjustable weights with live ranking; evidence panel linking this repo's measured runs. Content in `src/content/methods.ts`, labelled "team assessment". |
| `/compute` | Target cards with health, usage, quota; admins: create/edit (form from `/compute-targets/schema/{kind}`), set secret (write-only textarea), run health check |
| `/admin/users` | Better Auth admin API: list, create (name, email, temporary password, role), change role, ban/unban, set password, revoke sessions; each action also POSTs an audit event |
| `/admin/audit` | Audit event table with filters |

Client data: SSE via `EventSource('/api/backend/runs/<id>/events?after_log_id=…')` with reconnect using the last log id.
Types: `src/lib/api/schema.d.ts` is generated from the API's OpenAPI schema (`pnpm gen:api`; source `OPENAPI_SOURCE`,
default the running API, in CI `skf-api openapi`), and `src/lib/api/types.ts` only aliases its component schemas, so any
API/web drift fails `tsc`. CI regenerates the file and fails if it changed.

## 12. Engineering rules

- No business logic in routers or React components; services and hooks hold it.
- Every external system (S3, Redis, Kaggle, AWS, SSH, subprocess) sits behind an interface with a fake for tests.
- Settings only from typed env (`pydantic-settings`; web `src/lib/env.ts` with zod). No secrets in logs, responses or the browser.
- No shell strings: argv lists everywhere, including SSH remote commands (quote with `shlex.quote`).
- Every schema change has an Alembic migration; `skf-api migrate` runs `alembic upgrade head`.
- Never start real paid resources in tests or during development verification: no `aws ec2 start-instances`/`run-instances`,
  no `kaggle kernels push`. Use fakes, `moto`, and read-only calls.
- Tests: `apps/api/tests` with a real Postgres (`skf_test` database; fixtures create/drop the `app` schema) and fakeredis-free
  design (use the real Redis container, db 15). Web: `tsc --noEmit`, lint, and Playwright e2e in `apps/web/e2e`.
