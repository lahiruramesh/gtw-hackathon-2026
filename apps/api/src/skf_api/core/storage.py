"""Artifact storage on any S3 API (MinIO in dev, S3 in production).

boto3 is synchronous, so every call runs in a worker thread.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class PresignedUrl:
    url: str
    expires_at: datetime


class Storage(Protocol):
    async def ensure_bucket(self) -> None: ...
    async def upload_file(
        self, key: str, path: Path, *, content_type: str, tags: Mapping[str, str] | None = None
    ) -> None: ...
    async def download_file(self, key: str, path: Path) -> None: ...
    async def presigned_get(
        self, key: str, *, filename: str, content_type: str, ttl: timedelta = timedelta(minutes=15)
    ) -> PresignedUrl: ...


@dataclass(frozen=True)
class S3Config:
    bucket: str
    region: str
    endpoint_url: str | None = None
    public_endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None


class S3Storage:
    def __init__(self, config: S3Config):
        self.bucket = config.bucket
        self._region = config.region
        self._client = self._make_client(config, config.endpoint_url)
        # Presigning is offline, so a second client with the browser-facing host is free.
        self._public_client = self._make_client(config, config.public_endpoint_url or config.endpoint_url)

    @staticmethod
    def _make_client(config: S3Config, endpoint_url: str | None) -> Any:
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=config.region,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=Config(
                signature_version="s3v4",
                # MinIO serves buckets as paths. Real S3 URLs must name the bucket in the host
                # (https://<bucket>.s3.amazonaws.com/...): that is the origin the web app's CSP allows for
                # videos and images, and path-style URLs are blocked by it.
                s3={"addressing_style": "path" if endpoint_url else "virtual"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    async def ensure_bucket(self) -> None:
        def _ensure() -> None:
            try:
                self._client.head_bucket(Bucket=self.bucket)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket", "NotFound"):
                    raise
                kwargs: dict[str, Any] = {"Bucket": self.bucket}
                if self._region != "us-east-1":
                    kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self._region}
                self._client.create_bucket(**kwargs)

        await asyncio.to_thread(_ensure)

    async def upload_file(
        self, key: str, path: Path, *, content_type: str, tags: Mapping[str, str] | None = None
    ) -> None:
        """`tags` become S3 object tags; bucket lifecycle rules match on them (e.g. `skf-kind=log`)."""
        extra: dict[str, str] = {"ContentType": content_type}
        if tags:
            extra["Tagging"] = urlencode(tags)
        await asyncio.to_thread(self._client.upload_file, str(path), self.bucket, key, ExtraArgs=extra)

    async def download_file(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._client.download_file, self.bucket, key, str(path))

    async def presigned_get(
        self, key: str, *, filename: str, content_type: str, ttl: timedelta = timedelta(minutes=15)
    ) -> PresignedUrl:
        safe_name = "".join(c for c in filename if c.isascii() and c.isprintable() and c not in '"\\')
        params = {
            "Bucket": self.bucket,
            "Key": key,
            "ResponseContentType": content_type,
            "ResponseContentDisposition": f'inline; filename="{safe_name}"',
        }
        url = await asyncio.to_thread(
            self._public_client.generate_presigned_url,
            "get_object",
            Params=params,
            ExpiresIn=int(ttl.total_seconds()),
        )
        return PresignedUrl(url=url, expires_at=datetime.now(UTC) + ttl)
