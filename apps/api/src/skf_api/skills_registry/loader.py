"""Load `skills/*/skill.yaml` from the pipeline repo. A broken manifest is reported, never fatal."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from skf_api.skills_registry.manifest import Manifest


@dataclass(frozen=True)
class LoadedSkill:
    manifest: Manifest
    directory: Path
    manifest_sha: str


@dataclass(frozen=True)
class LoadError:
    file: str
    message: str


@dataclass(frozen=True)
class LoadResult:
    skills: dict[str, LoadedSkill] = field(default_factory=dict)
    errors: list[LoadError] = field(default_factory=list)


def _describe(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'manifest'}: {e['msg']}" for e in exc.errors())


def load_skills(skills_dir: Path) -> LoadResult:
    result = LoadResult()
    if not skills_dir.is_dir():
        result.errors.append(LoadError(str(skills_dir), "skills directory not found"))
        return result
    for path in sorted(skills_dir.glob("*/skill.yaml")):
        rel = str(path.relative_to(skills_dir.parent))
        raw = path.read_bytes()
        try:
            manifest = Manifest.model_validate(yaml.safe_load(raw))
        except yaml.YAMLError as exc:
            result.errors.append(LoadError(rel, f"invalid YAML: {exc}"))
            continue
        except ValidationError as exc:
            result.errors.append(LoadError(rel, _describe(exc)))
            continue
        if manifest.id in result.skills:
            result.errors.append(LoadError(rel, f"duplicate skill id '{manifest.id}'"))
            continue
        result.skills[manifest.id] = LoadedSkill(manifest, path.parent, hashlib.sha256(raw).hexdigest())
    return result
