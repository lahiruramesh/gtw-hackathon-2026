"""Fixtures for backend tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from backend_fakes import REPO_ROOT

from skf_api.backends.base import TargetContext


@pytest.fixture
def make_ctx(tmp_path: Path) -> Callable[..., TargetContext]:
    def make(
        config: dict[str, Any], secret: dict[str, Any] | None = None, repo: Path = REPO_ROOT
    ) -> TargetContext:
        return TargetContext(
            target_id="t-1",
            name="test target",
            config=config,
            secret=secret or {},
            work_dir=tmp_path / "work",
            pipeline_repo_dir=repo,
        )

    return make
