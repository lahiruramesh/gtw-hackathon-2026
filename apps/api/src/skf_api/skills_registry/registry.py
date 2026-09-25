"""In-process view of the skills in the pipeline repo (manifests plus their directories)."""

from __future__ import annotations

from pathlib import Path

from skf_api.skills_registry.loader import LoadedSkill, LoadResult, load_skills


class SkillRegistry:
    def __init__(self, skills_dir: Path, *, fallback_git_sha: str | None = None):
        self.skills_dir = skills_dir
        # Used when the repo has no .git (container images); see skills_registry.git.head_sha.
        self.fallback_git_sha = fallback_git_sha
        self._skills: dict[str, LoadedSkill] = {}

    def reload(self) -> LoadResult:
        result = load_skills(self.skills_dir)
        self._skills = dict(result.skills)
        return result

    def directory(self, skill_id: str) -> Path | None:
        loaded = self._skills.get(skill_id)
        return loaded.directory if loaded else None
