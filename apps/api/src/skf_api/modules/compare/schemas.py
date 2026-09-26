from __future__ import annotations

from pydantic import BaseModel

from skf_api.modules.runs.schemas import RunDetail
from skf_api.skills_registry.manifest import GateLevel


class HeadlineRow(BaseModel):
    label: str
    unit: str | None
    values: list[float | None]


class GateRow(BaseModel):
    label: str
    level: GateLevel
    values: list[bool | None]  # null: the run has no value for the criterion's metric (never measured)


class Comparison(BaseModel):
    runs: list[RunDetail]
    headline_rows: list[HeadlineRow]
    gate_rows: list[GateRow]
    metric_keys: list[str]
