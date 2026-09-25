"""Validate run params against a manifest: types, bounds, enums, nullability; unknown keys rejected."""

from __future__ import annotations

import math
import uuid
from typing import Any

from skf_api.skills_registry.manifest import Manifest, ParamSpec, ParamType


class ParamsError(Exception):
    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def _coerce(spec: ParamSpec, value: Any) -> Any:
    """Returns the normalised value or raises ValueError with a user-facing reason."""
    match spec.type:
        case ParamType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError("must be a boolean")
            return value
        case ParamType.INTEGER:
            # JSON clients may send 2e8 as a float; accept it only when it is integral.
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("must be an integer")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("must be an integer")
            return int(value)
        case ParamType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError("must be a number")
            return float(value)
        case ParamType.STRING:
            if not isinstance(value, str):
                raise ValueError("must be a string")
            return value
        case ParamType.CHECKPOINT:
            try:
                return str(uuid.UUID(str(value)))
            except ValueError:
                raise ValueError("must be a checkpoint artifact id") from None


def validate_params(manifest: Manifest, params: dict[str, Any]) -> dict[str, Any]:
    """Returns params with every manifest default applied (the canonical form stored on the run)."""
    errors: dict[str, str] = {name: "unknown parameter" for name in params if name not in manifest.params}
    result: dict[str, Any] = {}
    for name, spec in manifest.params.items():
        value = params.get(name, spec.default)
        if value is None:
            if not spec.nullable:
                errors[name] = "is required"
            result[name] = None
            continue
        try:
            value = _coerce(spec, value)
        except ValueError as exc:
            errors[name] = str(exc)
            continue
        if spec.minimum is not None and value < spec.minimum:
            errors[name] = f"must be >= {spec.minimum:g}"
        elif spec.maximum is not None and value > spec.maximum:
            errors[name] = f"must be <= {spec.maximum:g}"
        elif spec.enum is not None and value not in spec.enum:
            errors[name] = f"must be one of {', '.join(map(str, spec.enum))}"
        result[name] = value
    if errors:
        raise ParamsError(errors)
    return result


def checkpoint_params(manifest: Manifest, params: dict[str, Any]) -> dict[str, uuid.UUID]:
    """Checkpoint-typed params that are set, as artifact ids."""
    return {
        name: uuid.UUID(params[name])
        for name, spec in manifest.params.items()
        if spec.type is ParamType.CHECKPOINT and params.get(name)
    }
