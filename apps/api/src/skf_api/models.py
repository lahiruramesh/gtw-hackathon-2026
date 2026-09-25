"""Imports every ORM model so `Base.metadata` is complete (Alembic, mapper configuration)."""

from skf_api.core.db import Base
from skf_api.modules.artifacts.models import Artifact
from skf_api.modules.audit.models import AuditEvent
from skf_api.modules.compute.models import ComputeTarget
from skf_api.modules.gates.models import GateDecision
from skf_api.modules.runs.models import Evaluation, LogLine, MetricPoint, Run, Stage
from skf_api.modules.skills.models import Skill

__all__ = [
    "Artifact",
    "AuditEvent",
    "Base",
    "ComputeTarget",
    "Evaluation",
    "GateDecision",
    "LogLine",
    "MetricPoint",
    "Run",
    "Skill",
    "Stage",
]
