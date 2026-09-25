from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

MAX_LINES_PER_BATCH = 1000
MAX_BATCH_BYTES = 1_000_000


class IncomingLine(BaseModel):
    ts: datetime | None = None
    text: str
    stream: Literal["stdout", "stderr"] | None = None


class LogBatch(BaseModel):
    lines: list[IncomingLine] = Field(max_length=MAX_LINES_PER_BATCH)
    noise_dropped: int = Field(default=0, ge=0)


class MetricPointIn(BaseModel):
    step: int = Field(ge=0)
    wall_s: float | None = None
    values: dict[str, float]


class MetricBatch(BaseModel):
    points: list[MetricPointIn] = Field(max_length=MAX_LINES_PER_BATCH)


class Heartbeat(BaseModel):
    progress: float | None = Field(default=None, ge=0, le=1)
    message: str | None = Field(default=None, max_length=500)


class IngestAck(BaseModel):
    accepted: int
    dropped: int = 0
