"""Turn a finished job's outputs into artifacts, metrics and (for evaluate stages) an evaluation row."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from skf_api.context import AppContext
from skf_api.core.db import utcnow
from skf_api.core.storage import Storage
from skf_api.modules.artifacts.models import Artifact, artifact_key
from skf_api.modules.artifacts.service import classify, content_type
from skf_api.modules.ingest.progress_csv import read_progress_csv
from skf_api.modules.ingest.service import IngestService
from skf_api.modules.runs.models import Evaluation, MetricPoint, Run, Stage, StageKind
from skf_api.skills_registry.manifest import StageDef
from skf_api.skills_registry.summarize import SummarizeError, summarize


class CollectError(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_files(out_dir: Path) -> list[Path]:
    """Regular files under out_dir. Symlinks are skipped and symlinked directories are not entered: outputs
    come from a compute target, which is less trusted than the worker, and following a link such as
    `out/notes.txt -> /proc/self/environ` would publish a worker-host file as a downloadable artifact."""
    files = []
    for dirpath, _, filenames in os.walk(out_dir):  # followlinks=False
        for name in filenames:
            path = Path(dirpath) / name
            if not path.is_symlink() and path.is_file():
                files.append(path)
    return files


def remove_symlinks(out_dir: Path) -> None:
    """Metrics and the skill summarizer read files from out_dir too; none of them may read through a link."""
    for dirpath, dirnames, filenames in os.walk(out_dir):
        for name in [*dirnames, *filenames]:
            path = Path(dirpath) / name
            if path.is_symlink():
                path.unlink()


async def store_outputs(
    session: AsyncSession, storage: Storage, stage: Stage, out_dir: Path, files: list[Path] | None = None
) -> int:
    """Uploads files under out_dir (default: all regular files) to `runs/<run>/<stage key>/<relative path>`
    and upserts their artifact rows (idempotent: a repeated collect overwrites the same keys). Returns the
    file count."""
    if files is None:
        files = await asyncio.to_thread(output_files, out_dir)
    files = sorted(files)
    for path in files:
        name = path.relative_to(out_dir).as_posix()
        kind, step = classify(name)
        key = artifact_key(stage.run_id, stage.key, name)
        mime = content_type(name)
        sha = await asyncio.to_thread(_sha256, path)
        stored_sha = await session.scalar(
            sa.select(Artifact.sha256).where(Artifact.run_id == stage.run_id, Artifact.uri == key)
        )
        if stored_sha != sha:
            await storage.upload_file(key, path, content_type=mime, tags={"skf-kind": kind})
        values = {
            "run_id": stage.run_id,
            "stage_id": stage.id,
            "kind": kind,
            "name": name,
            "uri": key,
            "size_bytes": path.stat().st_size,
            "sha256": sha,
            "content_type": mime,
            "step": step,
            "created_at": utcnow(),
        }
        stmt = insert(Artifact).values(**values)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[Artifact.run_id, Artifact.uri],
                set_={
                    k: stmt.excluded[k]
                    for k in (
                        "stage_id",
                        "kind",
                        "name",
                        "size_bytes",
                        "sha256",
                        "content_type",
                        "step",
                        "created_at",
                    )
                },
            )
        )
    return len(files)


async def metrics_from_progress(
    ingest: IngestService, session: AsyncSession, stage: Stage, stage_def: StageDef, out_dir: Path
) -> None:
    """Fallback for jobs whose reporter could not phone home: load progress.csv once, after the fact."""
    if not stage_def.progress_csv:
        return
    path = out_dir / stage_def.progress_csv
    has_metrics = await session.scalar(
        sa.select(MetricPoint.id).where(MetricPoint.stage_id == stage.id).limit(1)
    )
    if path.is_file() and has_metrics is None:
        await ingest.write_metrics(session, stage, read_progress_csv(path), touch=False)


async def record_evaluation(
    session: AsyncSession, skill_dir: Path | None, run: Run, stage: Stage, stage_def: StageDef, out_dir: Path
) -> None:
    if stage.kind is not StageKind.EVALUATE:
        return
    if skill_dir is None:
        raise CollectError(
            f"Skill '{run.skill_id}' is not in the pipeline repo; cannot summarise the evaluation"
        )
    try:
        summary = await summarize(skill_dir, out_dir)
    except SummarizeError as exc:
        raise CollectError(str(exc)) from exc
    if summary is None:
        return
    await session.execute(sa.delete(Evaluation).where(Evaluation.stage_id == stage.id))
    session.add(
        Evaluation(
            run_id=run.id,
            stage_id=stage.id,
            checkpoint_id=stage.checkpoint_id,
            suite=stage_def.suite or stage_def.id,
            summary=summary,
            created_at=utcnow(),
        )
    )


async def collect_outputs(
    ctx: AppContext,
    session: AsyncSession,
    ingest: IngestService,
    run: Run,
    stage: Stage,
    stage_def: StageDef,
    out_dir: Path,
) -> int:
    """Everything after backend.collect() filled out_dir. The caller commits and marks the stage succeeded."""
    await asyncio.to_thread(remove_symlinks, out_dir)
    count = await store_outputs(session, ctx.storage, stage, out_dir)
    await session.flush()
    await metrics_from_progress(ingest, session, stage, stage_def, out_dir)
    await record_evaluation(session, ctx.registry.directory(run.skill_id), run, stage, stage_def, out_dir)
    return count
