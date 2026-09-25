# SKF Skill Studio — web

Next.js 16 (App Router) front end for the Skill Studio: sign-in (Better Auth), run wizard, live logs and
metrics, evaluation, release gate, approvals, compute targets and user management. The contract it builds
against is `docs/webapp/SPEC.md` (§4 auth, §9 API shapes, §11 pages).

## Run locally

```bash
cp .env.example .env.local        # fill BETTER_AUTH_SECRET and ADMIN_PASSWORD
pnpm install
pnpm auth:migrate                 # Better Auth tables in the `auth` schema
pnpm seed:admin                   # first admin from ADMIN_EMAIL / ADMIN_PASSWORD (idempotent)
pnpm dev -p 3100
```

The API (`apps/api`) must run on `API_INTERNAL_URL`; without it every page shows an "API is not reachable"
state instead of failing.

## Checks

| Command          | What                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `pnpm lint`      | ESLint (Next.js rules)                                                                                                       |
| `pnpm typecheck` | `tsc --noEmit`                                                                                                               |
| `pnpm test`      | Vitest unit tests for the pure modules in `src/lib`                                                                          |
| `pnpm test:e2e`  | Playwright against a running stack (`E2E_BASE_URL`, default `http://localhost:3100`)                                         |
| `pnpm gen:api`   | Regenerate `src/lib/api/schema.d.ts` from the API's OpenAPI (`OPENAPI_SOURCE`, default `http://localhost:8000/openapi.json`) |
| `pnpm format`    | Prettier (with Tailwind class ordering)                                                                                      |
| `pnpm build`     | Production build, `output: "standalone"`                                                                                     |

The e2e suite signs in as the seeded admin, creates (or re-keys) one `e2e-<role>@skf.local` account per role through
`/admin/users`, and keeps their sessions and rotated passwords in the gitignored `e2e/.auth/`. `e2e/launch.spec.ts`
launches and then cancels a real smoke run on Local CPU, so the worker must be running with `PIPELINE_PYTHON` set.

## Layout

| Path                                              | What                                                                                                      |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `src/app/(auth)/login`                            | Sign-in page (email + password; no sign-up)                                                               |
| `src/app/(app)/*`                                 | Signed-in pages; each fetches on the server with `apiFetch`                                               |
| `src/app/api/auth/[...all]`                       | Better Auth handler (sessions, JWKS for the API)                                                          |
| `src/app/api/backend/[...path]`                   | Same-origin proxy to `API_INTERNAL_URL/api/v1/*` for client components and SSE                            |
| `src/lib/api/`                                    | Typed API client (`types.ts` aliases the generated `schema.d.ts`), errors, server fetch, proxy path rules |
| `src/lib/auth.ts`, `session.ts`, `permissions.ts` | Better Auth config, viewer lookup, roles from `shared/permissions.json`                                   |
| `src/components/`                                 | UI by feature (`runs/`, `compute/`, `admin/` …); `ui/` is shadcn/ui                                       |
| `src/content/methods.ts`                          | Learning-methods matrix and project evidence (team assessment)                                            |

## Deployment notes

- `next build` needs no runtime secrets: env is validated on first use (`src/lib/env.ts`).
- Build from the repository root context: the app imports `shared/permissions.json`, so tracing starts at
  the repo root and the standalone server is `.next/standalone/apps/web/server.js`. Copy `.next/static`
  to `.next/standalone/apps/web/.next/static` and `public` to `.next/standalone/apps/web/public`.
- The repository's root `.gitignore` ignores every `runs/` directory; `apps/web/.gitignore` re-includes
  `src/**/runs/`, and `globals.css` registers `src` with `@source` so Tailwind scans those files.
