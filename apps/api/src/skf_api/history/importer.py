"""`skf-api import-history`: past experiments become imported runs, shaped exactly like live ones.

For each experiment: a run with train / evaluate / gate stages, train artifacts and metrics from its output
directory, evaluation artifacts and summaries produced by the skill's own summarizer (so gate metrics resolve
the same way as for a live run), one side stage per strict-tested checkpoint, then the release gate.
Re-running is safe: rows are matched by run name and stage key, artifacts by storage key, and the gate is
evaluated again.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.backends.base import BackendKind
from skf_api.context import AppContext
from skf_api.core.errors import NotFound
from skf_api.history.catalog import EXPERIMENTS, Experiment, StairsResults, StepLengthResults
from skf_api.history.effort_log import EffortLog
from skf_api.modules.artifacts.models import Artifact, artifact_key
from skf_api.modules.audit import service as audit
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.gates.service import evaluate_gate
from skf_api.modules.ingest.progress_csv import read_progress_csv
from skf_api.modules.ingest.service import IngestService, Point
from skf_api.modules.runs import transitions
from skf_api.modules.runs.models import Run, RunsOn, RunStatus, Stage, StageKind, StageStatus
from skf_api.modules.skills.service import get_skill
from skf_api.orchestrator.collector import record_evaluation, store_outputs
from skf_api.skills_registry.manifest import Manifest, StageDef
from skf_api.skills_registry.params import validate_params

IMPORT_ACTOR = audit.Actor(id="system:import", name="History import", role="system")
_CKPT_RESULT = re.compile(r"_ckpt_(\d+)\.json$")
_INIT_FROM = re.compile(r"(?:^|/)runs/([^/]+)/run/([^/]+)$")
TRAIN_FILES = ("params.pkl", "config.json", "progress.csv")
STRICT_FILE = "strict.json"
VIDEO_FILE = "crossing.mp4"


class SkipExperiment(Exception):
    """An experiment cannot be imported (e.g. no seeded target); reported, and the others continue."""


@dataclass(frozen=True)
class Imported:
    experiment: Experiment
    run: Run
    init_from: str | None  # config.json's warm-start path, e.g. runs/g1-stairs-v10/run/ckpt_00253624320.pkl


def _params_from_config(manifest: Manifest, config: dict[str, Any]) -> dict[str, Any]:
    """The manifest params this run was trained with, read back from g1pipe.train's config.json."""
    ppo, env = config.get("ppo", {}), config.get("env", {})
    known = {
        "timesteps": ppo.get("num_timesteps"),
        "seed": config.get("seed"),
        "lr": ppo.get("learning_rate"),
        "no_dr": config.get("no_dr"),
        "scan_model": env.get("scan_model"),
        "leg_action_scale": env.get("leg_action_scale") or None,
    }
    return validate_params(
        manifest, {k: v for k, v in known.items() if k in manifest.params and v is not None}
    )


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


class HistoryImporter:
    def __init__(
        self,
        ctx: AppContext,
        results_dir: Path,
        runs_dir: Path,
        *,
        with_checkpoints: bool,
        echo: Callable[[str], None],
    ):
        self._ctx = ctx
        self._results = results_dir
        self._runs = runs_dir
        self._with_checkpoints = with_checkpoints
        self._echo = echo
        self._effort = EffortLog.read(results_dir.parent / "docs" / "effort_log.csv")
        self._ingest = IngestService(ctx.redis, ctx.events, ctx.settings.log_max_lines_per_run)

    async def run(self, experiments: tuple[Experiment, ...] = EXPERIMENTS) -> list[Imported]:
        imported: list[Imported] = []
        for exp in experiments:
            async with self._ctx.db.session() as session:
                try:
                    done = await self._import(session, exp)
                except (SkipExperiment, NotFound) as exc:
                    await session.rollback()
                    self._echo(f"skip {exp.name}: {exc}")
                    continue
            if done is not None:
                imported.append(done)
        async with self._ctx.db.session() as session:
            await self._link_parents(session, imported)
        return imported

    # ------------------------------------------------------------------------------------------ one run

    async def _import(self, session: AsyncSession, exp: Experiment) -> Imported | None:
        train_dir = self._runs / exp.name / "run"
        if not (train_dir / "params.pkl").is_file():
            self._echo(f"skip {exp.name}: no {train_dir}/params.pkl")
            return None
        _, manifest = await get_skill(session, exp.skill_id)
        target = await self._target(session, name=exp.target)
        local = await self._target(session, kind=BackendKind.LOCAL_CPU)
        existing = await session.scalar(sa.select(Run).where(Run.name == exp.name))
        if existing is not None and not existing.imported:
            raise SkipExperiment("a live run already uses this name")

        progress = read_progress_csv(train_dir / "progress.csv")
        finished = _mtime(train_dir / "params.pkl")
        wall_s = max((p.wall_s or 0.0 for p in progress), default=0.0)
        started = finished - timedelta(seconds=wall_s)
        gpu_hours = self._effort.gpu_hours(exp.effort_match) or wall_s / 3600.0
        config = json.loads((train_dir / "config.json").read_text())

        run = existing or self._new_run(exp, manifest, config, target, started)
        if existing is None:
            session.add(run)
            await session.flush()
            audit.record(session, IMPORT_ACTOR, "run.import", "run", run.id, {"name": exp.name})
        stages = await self._stages(session, run, manifest, target, local)
        train, evaluate, gate = stages["train"], stages["evaluate"], stages["gate"]
        for stage in (train, evaluate, gate):
            stage.status = StageStatus.SUCCEEDED
            stage.started_at, stage.finished_at = (
                (started, finished) if stage is train else (finished, finished)
            )
        train.gpu_seconds = gpu_hours * 3600.0
        train.cost = round(gpu_hours * target.cost_per_gpu_hour, 4)
        train.progress = evaluate.progress = 1.0
        run.started_at, run.finished_at = started, finished

        checkpoint_results = self._checkpoint_results(exp)
        await self._train_outputs(session, train, train_dir, progress, set(checkpoint_results))
        eval_def = manifest.evaluate_stages[0]
        await self._final_evaluation(session, exp, run, evaluate, eval_def)
        for step, result in sorted(checkpoint_results.items()):
            await self._checkpoint_evaluation(session, run, eval_def, local, step, result, finished)

        run.status = RunStatus.RUNNING  # evaluate_gate sets the final status
        decision = await evaluate_gate(session, run, manifest)
        passed = sum(c["passed"] for c in decision.criteria)
        gate.message = f"{decision.verdict.value}: {passed}/{len(decision.criteria)} criteria met"
        await session.commit()
        self._echo(
            f"imported {exp.name}: {len(checkpoint_results)} checkpoint evaluations, "
            f"{gpu_hours:.2f} GPU-h, gate {decision.verdict.value}"
        )
        return Imported(exp, run, config.get("init_from"))

    async def _target(
        self, session: AsyncSession, *, name: str | None = None, kind: BackendKind | None = None
    ) -> ComputeTarget:
        stmt = sa.select(ComputeTarget).order_by(ComputeTarget.created_at).limit(1)
        stmt = stmt.where(ComputeTarget.name == name) if name else stmt.where(ComputeTarget.kind == kind)
        target = await session.scalar(stmt)
        if target is None:
            raise SkipExperiment(
                f"compute target {name or kind} not found (run `skf-api seed-targets` first)"
            )
        return target

    def _new_run(
        self,
        exp: Experiment,
        manifest: Manifest,
        config: dict[str, Any],
        target: ComputeTarget,
        started: datetime,
    ) -> Run:
        return Run(
            name=exp.name,
            skill_id=exp.skill_id,
            preset_id=exp.preset_id,
            params=_params_from_config(manifest, config),
            status=RunStatus.RUNNING,
            compute_target_id=target.id,
            created_by_id=IMPORT_ACTOR.id,
            created_by_name=IMPORT_ACTOR.name,
            created_by_role=IMPORT_ACTOR.role,
            notes=exp.notes,
            imported=True,
            created_at=started,
        )

    async def _stages(
        self, session: AsyncSession, run: Run, manifest: Manifest, target: ComputeTarget, local: ComputeTarget
    ) -> dict[str, Stage]:
        existing = {
            s.key: s
            for s in await session.scalars(
                sa.select(Stage).where(Stage.run_id == run.id, Stage.checkpoint_id.is_(None))
            )
        }
        for position, stage_def in enumerate(manifest.pipeline):
            if stage_def.id in existing:
                continue
            target_id = (
                None
                if stage_def.kind is StageKind.GATE
                else (target.id if stage_def.runs_on is RunsOn.TARGET else local.id)
            )
            stage = Stage(
                run_id=run.id,
                key=stage_def.id,
                kind=stage_def.kind,
                title=stage_def.title,
                position=position,
                runs_on=stage_def.runs_on,
                status=StageStatus.PENDING,
                compute_target_id=target_id,
                attempt=1,
            )
            session.add(stage)
            existing[stage_def.id] = stage
        await session.flush()
        by_kind = {s.kind: s for s in existing.values()}
        return {
            "train": by_kind[StageKind.TRAIN],
            "evaluate": by_kind[StageKind.EVALUATE],
            "gate": by_kind[StageKind.GATE],
        }

    # ------------------------------------------------------------------------------------------ evidence

    def _checkpoint_results(self, exp: Experiment) -> dict[int, Path]:
        if not isinstance(exp.results, StairsResults):
            return {}
        pattern = f"strict_v{exp.results.version}_ckpt_*.json"
        return {
            int(m.group(1)): p
            for p in (self._results / "stairs").glob(pattern)
            if (m := _CKPT_RESULT.search(p.name))
        }

    async def _train_outputs(
        self,
        session: AsyncSession,
        train: Stage,
        train_dir: Path,
        progress: list[Point],
        evaluated_steps: set[int],
    ) -> None:
        files = self._train_files(train_dir, evaluated_steps)
        await store_outputs(session, self._ctx.storage, train, train_dir, files)
        await self._ingest.write_metrics(session, train, progress, touch=False)

    def _train_files(self, train_dir: Path, evaluated_steps: set[int]) -> list[Path]:
        files = [train_dir / name for name in TRAIN_FILES if (train_dir / name).is_file()]
        for ckpt in sorted(train_dir.glob("ckpt_*.pkl")):
            if self._with_checkpoints or int(ckpt.stem.split("_")[1]) in evaluated_steps:
                files.append(ckpt)
        return files

    async def _final_evaluation(
        self, session: AsyncSession, exp: Experiment, run: Run, stage: Stage, stage_def: StageDef
    ) -> None:
        if isinstance(exp.results, StepLengthResults):
            await self._evaluation(session, run, stage, stage_def, self._results / exp.results.directory)
            return
        strict = self._results / "stairs" / f"strict_v{exp.results.version}_params.json"
        video = self._results / "videos" / exp.results.video if exp.results.video else None
        with tempfile.TemporaryDirectory(prefix="skf-import-") as tmp:
            out = Path(tmp)
            if strict.is_file():
                shutil.copyfile(strict, out / STRICT_FILE)
            if video and video.is_file():
                shutil.copyfile(video, out / VIDEO_FILE)
            if not strict.is_file():
                stage.message = "The final policy was not strict-tested; see the checkpoint evaluations"
                await store_outputs(session, self._ctx.storage, stage, out)
                return
            await self._evaluation(session, run, stage, stage_def, out)

    async def _checkpoint_evaluation(
        self,
        session: AsyncSession,
        run: Run,
        stage_def: StageDef,
        local: ComputeTarget,
        step: int,
        result: Path,
        at: datetime,
    ) -> None:
        ckpt = await session.scalar(
            sa.select(Artifact).where(
                Artifact.run_id == run.id, Artifact.step == step, Artifact.uri.like(f"runs/{run.id}/train/%")
            )
        )
        if ckpt is None:
            self._echo(f"  {run.name}: no ckpt for step {step}; its strict test is skipped")
            return
        key = transitions.side_stage_key(stage_def.id, ckpt)
        stage = await session.scalar(sa.select(Stage).where(Stage.run_id == run.id, Stage.key == key))
        if stage is None:
            position = await session.scalar(
                sa.select(sa.func.max(Stage.position)).where(Stage.run_id == run.id)
            )
            stage = Stage(
                run_id=run.id,
                key=key,
                kind=StageKind.EVALUATE,
                title=f"{stage_def.title} @ {ckpt.name}",
                position=(position or 0) + 1,
                runs_on=RunsOn.LOCAL,
                status=StageStatus.SUCCEEDED,
                compute_target_id=local.id,
                attempt=1,
                checkpoint_id=ckpt.id,
            )
            session.add(stage)
            await session.flush()
        stage.started_at = stage.finished_at = at
        stage.progress = 1.0
        with tempfile.TemporaryDirectory(prefix="skf-import-") as tmp:
            shutil.copyfile(result, Path(tmp) / STRICT_FILE)
            await self._evaluation(session, run, stage, stage_def, Path(tmp))

    async def _evaluation(
        self, session: AsyncSession, run: Run, stage: Stage, stage_def: StageDef, out: Path
    ) -> None:
        await store_outputs(session, self._ctx.storage, stage, out)
        await record_evaluation(
            session, self._ctx.registry.directory(run.skill_id), run, stage, stage_def, out
        )
        await session.flush()

    # ------------------------------------------------------------------------------------------ lineage

    async def _link_parents(self, session: AsyncSession, imported: list[Imported]) -> None:
        """config.json `init_from: runs/<parent>/run/<file>` -> parent run and warm-start checkpoint."""
        for item in imported:
            match = _INIT_FROM.search(item.init_from or "")
            if not match:
                continue
            parent_name, filename = match.groups()
            parent = await session.scalar(sa.select(Run).where(Run.name == parent_name))
            if parent is None:
                self._echo(f"  {item.run.name}: parent {parent_name} is not imported; lineage left empty")
                continue
            ckpt = await session.scalar(
                sa.select(Artifact).where(
                    Artifact.run_id == parent.id, Artifact.uri == artifact_key(parent.id, "train", filename)
                )
            )
            run = await session.get(Run, item.run.id)
            assert run is not None
            run.parent_run_id = parent.id
            run.parent_checkpoint_id = ckpt.id if ckpt else None
            if ckpt is not None:
                run.params = {**run.params, "init_from": str(ckpt.id)}
            self._echo(f"  {run.name} <- {parent_name}{f' @ {filename}' if ckpt else ''}")
        await session.commit()
