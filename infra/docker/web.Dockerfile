# syntax=docker/dockerfile:1.7
# SKF Skill Studio web (Next.js standalone). Build from the repo root:
#   docker build -f infra/docker/web.Dockerfile -t skf-studio/web .                   # server (default)
#   docker build -f infra/docker/web.Dockerfile --target tools -t skf-studio/web-tools .
#
# `tools` keeps the full dependency tree and sources for one-off package scripts that the trimmed
# standalone server can't run: `pnpm auth:migrate` (Better Auth schema) and `pnpm seed:admin`.
# Both run as the base image's unprivileged `node` user (uid 1000).
# The repo layout is kept (/app/apps/web next to /app/shared) because the web app imports
# shared/permissions.json from outside its own directory.

ARG NODE_VERSION=24

FROM node:${NODE_VERSION}-slim AS base
ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH \
    COREPACK_HOME=/corepack \
    COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    NEXT_TELEMETRY_DISABLED=1
RUN corepack enable
WORKDIR /app/apps/web

FROM base AS deps
COPY apps/web/package.json apps/web/pnpm-lock.yaml apps/web/pnpm-workspace.yaml ./
# pnpm at the exact version pinned by package.json#packageManager.
RUN corepack install
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --store-dir /pnpm/store

FROM deps AS build
COPY shared/ /app/shared/
COPY apps/web/ ./
# No secrets at build time: src/lib/env.ts validates the environment lazily, when the server first needs it.
RUN pnpm build

FROM deps AS tools
COPY shared/ /app/shared/
COPY apps/web/ ./
USER 1000:1000
CMD ["pnpm", "auth:migrate"]

FROM node:${NODE_VERSION}-slim AS runtime
ARG GIT_SHA=
LABEL org.opencontainers.image.title="skf-studio-web" \
      org.opencontainers.image.revision="${GIT_SHA}"
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
# next.config.ts traces from the repo root, so the server lands at .next/standalone/apps/web/server.js.
WORKDIR /app
COPY --from=build /app/apps/web/.next/standalone ./
COPY --from=build /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=build /app/apps/web/public ./apps/web/public
WORKDIR /app/apps/web
USER 1000:1000
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["node", "-e", "fetch('http://127.0.0.1:3000/api/auth/ok').then(r => process.exit(r.ok ? 0 : 1), () => process.exit(1))"]
CMD ["node", "server.js"]
