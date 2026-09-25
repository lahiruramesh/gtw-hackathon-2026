"""Compute backends. `create_backend` is the only entry point the orchestrator uses."""

from __future__ import annotations

from skf_api.backends.base import BackendKind, ComputeBackend, TargetContext


def create_backend(kind: BackendKind | str, ctx: TargetContext) -> ComputeBackend:
    kind = BackendKind(kind)
    if kind is BackendKind.LOCAL_CPU:
        from skf_api.backends.local import LocalCpuBackend

        return LocalCpuBackend(ctx)
    if kind is BackendKind.KAGGLE:
        from skf_api.backends.kaggle import KaggleBackend

        return KaggleBackend(ctx)
    if kind is BackendKind.AWS_EC2:
        from skf_api.backends.aws_ec2 import AwsEc2Backend

        return AwsEc2Backend(ctx)
    raise ValueError(f"unknown backend kind {kind}")


def config_schema(kind: BackendKind | str) -> dict:
    """JSON Schema of a backend's non-secret config and secret, for the compute target form."""
    kind = BackendKind(kind)
    if kind is BackendKind.LOCAL_CPU:
        from skf_api.backends.local import LocalCpuConfig as C
        from skf_api.backends.local import LocalCpuSecret as S
    elif kind is BackendKind.KAGGLE:
        from skf_api.backends.kaggle import KaggleConfig as C
        from skf_api.backends.kaggle import KaggleSecret as S
    else:
        from skf_api.backends.aws_ec2 import AwsEc2Config as C
        from skf_api.backends.aws_ec2 import AwsEc2Secret as S
    return {"config": C.model_json_schema(), "secret": S.model_json_schema()}
