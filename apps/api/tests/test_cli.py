"""`skf-api openapi`: the schema the web app generates its TypeScript types from."""

from __future__ import annotations

import json

import pytest

from skf_api.cli import main


def test_openapi_prints_the_app_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["openapi"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "/api/v1/runs/{run_id}" in schema["paths"]
    summary = schema["components"]["schemas"]["RunSummary"]
    assert "parent" in summary["required"]
    assert schema["components"]["schemas"]["GateOp"]["enum"] == ["<", "<=", ">", ">=", "==", "!="]
