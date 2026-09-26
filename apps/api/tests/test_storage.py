"""Presigned artifact URLs must use the host the web app's CSP allows (media-src / img-src)."""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from skf_api.core.storage import S3Config, S3Storage


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")  # presigning is offline


async def presigned_host_and_path(config: S3Config) -> tuple[str, str]:
    url = (
        await S3Storage(config).presigned_get(
            "runs/r/evaluate/crossing.mp4", filename="crossing.mp4", content_type="video/mp4"
        )
    ).url
    parts = urlsplit(url)
    return parts.netloc, parts.path


async def test_real_s3_urls_name_the_bucket_in_the_host() -> None:
    host, path = await presigned_host_and_path(S3Config(bucket="skf-artifacts-123", region="us-east-1"))
    assert host == "skf-artifacts-123.s3.amazonaws.com"
    assert path == "/runs/r/evaluate/crossing.mp4"


async def test_minio_urls_stay_path_style() -> None:
    config = S3Config(
        bucket="skf-artifacts",
        region="eu-north-1",
        endpoint_url="http://minio:9000",
        public_endpoint_url="http://localhost:59000",
    )
    host, path = await presigned_host_and_path(config)
    assert host == "localhost:59000" and path.startswith("/skf-artifacts/runs/")
