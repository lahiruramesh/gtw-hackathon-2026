"""Which metric keys a skill stores (SPEC §5: default `eval/*`, without `*_std` and per-term reward
breakdowns `eval/episode_reward/*` unless a manifest lists them explicitly)."""

from __future__ import annotations

from fnmatch import fnmatchcase

from skf_api.skills_registry.manifest import MetricsSpec

_PER_TERM_PREFIX = "eval/episode_reward/"


class MetricFilter:
    def __init__(self, spec: MetricsSpec):
        self._exact = {k for k in spec.keys if not any(c in k for c in "*?[")}
        self._globs = [k for k in spec.keys if k not in self._exact]

    def accepts(self, key: str) -> bool:
        if key in self._exact:
            return True
        globs = [g for g in self._globs if fnmatchcase(key, g)]
        # A broad glob like `eval/*` does not opt into these families; a glob naming them does.
        if key.endswith("_std"):
            globs = [g for g in globs if "_std" in g]
        if key.startswith(_PER_TERM_PREFIX):
            globs = [g for g in globs if _PER_TERM_PREFIX in g]
        return bool(globs)
