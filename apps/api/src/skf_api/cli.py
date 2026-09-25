"""`skf-api` command line: migrate, sync-skills, seed-targets, import-history, openapi."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from alembic.config import Config

from alembic import command
from skf_api.context import AppContext, build_context
from skf_api.history.importer import HistoryImporter
from skf_api.modules.compute.seeds import default_seeds, seed_targets
from skf_api.modules.skills.service import sync_skills
from skf_api.settings import get_settings

API_DIR = Path(__file__).resolve().parents[2]  # apps/api (alembic.ini, alembic/)


def migrate(revision: str = "head") -> None:
    config = Config(str(API_DIR / "alembic.ini"))
    config.attributes["database_url"] = get_settings().database_url
    command.upgrade(config, revision)


async def _with_context(action: Callable[[AppContext], Awaitable[int]]) -> int:
    ctx = await build_context(get_settings())
    try:
        return await action(ctx)
    finally:
        await ctx.close()


async def _sync_skills(ctx: AppContext) -> int:
    async with ctx.db.session() as session:
        result = await sync_skills(session, ctx.registry)
    for skill_id in result.synced:
        print(f"synced {skill_id}")
    for error in result.errors:
        print(f"error {error.file}: {error.message}", file=sys.stderr)
    return 1 if result.errors else 0


def _seed_targets(kaggle_username: str | None) -> Callable[[AppContext], Awaitable[int]]:
    async def run(ctx: AppContext) -> int:
        async with ctx.db.session() as session:
            created = await seed_targets(session, default_seeds(kaggle_username))
        print("created " + ", ".join(created) if created else "all targets already exist")
        return 0

    return run


def _import_history(
    results_dir: Path, runs_dir: Path, with_checkpoints: bool
) -> Callable[[AppContext], Awaitable[int]]:
    async def run(ctx: AppContext) -> int:
        await ctx.storage.ensure_bucket()
        importer = HistoryImporter(ctx, results_dir, runs_dir, with_checkpoints=with_checkpoints, echo=print)
        imported = await importer.run()
        print(f"{len(imported)} runs imported")
        return 0

    return run


def _print_openapi() -> int:
    """The schema the web app's TypeScript types are generated from (`pnpm gen:api`); needs no services."""
    from skf_api.main import create_app

    json.dump(create_app().openapi(), sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skf-api", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="apply database migrations (alembic upgrade head)")
    commands.add_parser("sync-skills", help="load skills/*/skill.yaml into the catalog")
    seed = commands.add_parser("seed-targets", help="create the Local CPU, Kaggle T4 and AWS L40S targets")
    seed.add_argument("--kaggle-username", help="Kaggle account for the Kaggle T4 target")
    history = commands.add_parser("import-history", help="import past experiments as runs")
    history.add_argument("--results-dir", type=Path, required=True, help="the repo's results/ directory")
    history.add_argument(
        "--runs-dir", type=Path, required=True, help="directory holding <run>/run/params.pkl"
    )
    history.add_argument(
        "--with-checkpoints",
        action="store_true",
        help="upload every ckpt_*.pkl (default: only strict-tested ones)",
    )
    commands.add_parser("openapi", help="print the OpenAPI schema as JSON")
    args = parser.parse_args(argv)

    if args.command == "openapi":
        return _print_openapi()
    if args.command == "migrate":
        migrate()
        return 0
    if args.command == "sync-skills":
        action = _sync_skills
    elif args.command == "seed-targets":
        action = _seed_targets(args.kaggle_username)
    else:
        action = _import_history(args.results_dir.resolve(), args.runs_dir.resolve(), args.with_checkpoints)
    return asyncio.run(_with_context(action))


if __name__ == "__main__":
    sys.exit(main())
