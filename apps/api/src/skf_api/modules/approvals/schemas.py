from __future__ import annotations

from pydantic import BaseModel

from skf_api.modules.runs.schemas import RunSummary


class Approvals(BaseModel):
    launches: list[RunSummary]
    releases: list[RunSummary]
