"""Run a skill's `summarize.py` (`summarize(out_dir: Path) -> dict`), imported by file path."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

SUMMARIZER_FILE = "summarize.py"


class SummarizeError(Exception):
    pass


def load_summarizer(skill_dir: Path) -> Callable[[Path], dict[str, Any]] | None:
    path = skill_dir / SUMMARIZER_FILE
    if not path.is_file():
        return None
    # A unique module name per skill dir: two skills' summarizers must not share a sys.modules slot.
    spec = importlib.util.spec_from_file_location(f"skf_skill_summarizer_{skill_dir.name}", path)
    if spec is None or spec.loader is None:
        raise SummarizeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, "summarize", None)
    if not callable(func):
        raise SummarizeError(f"{path} does not define summarize(out_dir)")
    return cast(Callable[[Path], dict[str, Any]], func)


def _summarize_sync(skill_dir: Path, out_dir: Path) -> dict[str, Any] | None:
    func = load_summarizer(skill_dir)
    if func is None:
        return None
    try:
        summary = func(out_dir)
    except Exception as exc:
        raise SummarizeError(f"summarizer failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(summary, dict):
        raise SummarizeError("summarizer must return a dict")
    try:
        return json.loads(json.dumps(_finite(summary), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SummarizeError(f"summary is not plain JSON: {exc}") from exc


def _finite(value: Any) -> Any:
    """NaN/inf -> None: JSONB cannot store them, and a missing metric already fails its gate criterion."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


async def summarize(skill_dir: Path, out_dir: Path) -> dict[str, Any] | None:
    """None when the skill has no summarizer."""
    return await asyncio.to_thread(_summarize_sync, skill_dir, out_dir)
