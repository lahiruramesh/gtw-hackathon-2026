# SKF Skill Studio developer entry points. `make help` lists them; docs/webapp/RUNBOOK.md explains the flow.
#
# Dev loop: infra (postgres, redis, S3) in Docker, api/worker/web on the host for fast reloads:
#   make infra-up api-install web-install migrate seed    once
#   make api | make worker | make web                     one terminal each
# Everything in containers instead: make up

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

DOCKER ?= docker
# S3 service for local dev: minio (default) or rustfs, a drop-in for machines that can't pull minio/minio.
S3 ?= minio
COMPOSE := $(DOCKER) compose -f infra/compose.yaml $(if $(filter rustfs,$(S3)),-f infra/compose.rustfs.yaml)

# Past training outputs to import as historical runs: a checkout's runs/ (<run>/run/params.pkl) and results/.
HISTORY_DIR ?= runs
RESULTS_DIR ?= results
# Optional Kaggle account for the seeded "Kaggle T4" target.
KAGGLE_USERNAME ?=

# Image builds: `make build-images`; `make push-images REGISTRY=<acct>.dkr.ecr.<region>.amazonaws.com`.
GIT_SHA := $(shell git rev-parse HEAD)
TAG ?= $(shell git rev-parse --short=12 HEAD)
REGISTRY ?=
PLATFORM ?= linux/amd64
IMAGES := api:infra/docker/api.Dockerfile:runtime web:infra/docker/web.Dockerfile:runtime \
          web-tools:infra/docker/web.Dockerfile:tools worker:infra/docker/worker.Dockerfile:runtime

API := cd apps/api &&
WEB := cd apps/web &&
SHELL_SCRIPTS := infra/aws/*.sh infra/aws/host/*.sh

.PHONY: help
help: ## List targets
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

## --- local services --------------------------------------------------------------------------

.PHONY: infra-up
infra-up: ## Start postgres, redis and S3 in Docker (S3=rustfs if minio/minio can't be pulled)
	$(COMPOSE) up -d --wait postgres redis minio

.PHONY: infra-down
infra-down: ## Stop all dev containers, including the `make up` stack (data volumes are kept)
	$(COMPOSE) --profile app down

## --- API and worker (host) -------------------------------------------------------------------

.PHONY: api-install
api-install: ## Install API deps (apps/api/.venv); the pipeline itself needs `uv sync --extra train` at the root
	$(API) uv sync

.PHONY: migrate
migrate: ## Apply database migrations: app schema (Alembic) and auth schema (Better Auth)
	$(API) uv run skf-api migrate
	$(WEB) pnpm auth:migrate

.PHONY: api
api: ## Run the API on :8000 with reload
	$(API) uv run uvicorn skf_api.main:app --reload --port 8000

.PHONY: worker
worker: ## Run the arq worker (orchestrator), restarting on code changes
	@# arq's own --watch restarts the worker inside the same interpreter, so edited modules are never re-imported.
	@# Local jobs run in their own session and survive the restart.
	$(API) uv run watchfiles --filter python 'arq skf_api.orchestrator.tasks.WorkerSettings' src

## --- web (host) ------------------------------------------------------------------------------

.PHONY: web-install
web-install: ## Install web deps (pnpm)
	$(WEB) pnpm install --frozen-lockfile

.PHONY: web
web: ## Run Next.js on :3100
	$(WEB) pnpm dev -p 3100

## --- data ------------------------------------------------------------------------------------

.PHONY: seed
seed: ## Load skills/, create the Local CPU / Kaggle T4 / AWS L40S targets and the first admin (idempotent)
	$(API) uv run skf-api sync-skills
	$(API) uv run skf-api seed-targets $(if $(KAGGLE_USERNAME),--kaggle-username $(KAGGLE_USERNAME))
	$(WEB) pnpm seed:admin

.PHONY: import-history
import-history: ## Import past experiments as runs (HISTORY_DIR=<checkout>/runs, RESULTS_DIR=<checkout>/results)
	$(API) uv run skf-api import-history --results-dir $(abspath $(RESULTS_DIR)) --runs-dir $(abspath $(HISTORY_DIR))

## --- quality ---------------------------------------------------------------------------------

.PHONY: test
test: ## Run API tests (needs `make infra-up`) and web unit tests
	$(API) uv run pytest
	$(WEB) pnpm test

.PHONY: lint
lint: ## Lint and type-check API, web and infra
	$(API) uv run ruff check . && uv run ruff format --check . && uv run pyright
	$(WEB) pnpm lint && pnpm typecheck
	uvx --from shellcheck-py shellcheck -x $(SHELL_SCRIPTS)
	for f in infra/docker/*.Dockerfile; do uvx --from hadolint-bin hadolint "$$f"; done
	$(COMPOSE) config -q

.PHONY: fmt
fmt: ## Format API (ruff) and fix web lint issues (eslint --fix)
	$(API) uv run ruff format . && uv run ruff check --fix .
	$(WEB) pnpm lint --fix

## --- containers ------------------------------------------------------------------------------

.PHONY: build-images
build-images: ## Build the api, web, web-tools and worker images for this machine, tagged $(TAG)
	@for spec in $(IMAGES); do \
	  IFS=: read -r name file target <<<"$$spec"; \
	  echo "==> skf-studio/$$name:$(TAG)"; \
	  $(DOCKER) build -f "$$file" --target "$$target" --build-arg GIT_SHA=$(GIT_SHA) -t "skf-studio/$$name:$(TAG)" .; \
	done

.PHONY: push-images
push-images: ## Build for $(PLATFORM) and push to $(REGISTRY)/skf-studio/*:$(TAG) (log in to ECR first)
	@test -n "$(REGISTRY)" || { echo "set REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com" >&2; exit 2; }
	@for spec in $(IMAGES); do \
	  IFS=: read -r name file target <<<"$$spec"; \
	  echo "==> $(REGISTRY)/skf-studio/$$name:$(TAG)"; \
	  $(DOCKER) buildx build --platform $(PLATFORM) -f "$$file" --target "$$target" --build-arg GIT_SHA=$(GIT_SHA) \
	    -t "$(REGISTRY)/skf-studio/$$name:$(TAG)" --push .; \
	done

.PHONY: up
up: ## Build and run everything in Docker (needs .env from .env.example): web :3100, api :8000
	@test -f .env || { echo "copy .env.example to .env and fill in the secrets first" >&2; exit 2; }
	$(COMPOSE) --profile app up -d --build --wait

.PHONY: down
down: infra-down ## Stop the containerised stack (volumes are kept)
