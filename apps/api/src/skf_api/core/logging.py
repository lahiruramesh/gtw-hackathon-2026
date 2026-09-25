"""Structured JSON logging for the API and the worker."""

from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

import orjson
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Attributes every LogRecord has; anything else was passed via `extra=` and is emitted as a field.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return orjson.dumps(entry, default=str).decode()


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # uvicorn and the arq CLI install their own text handlers before the app starts: route them through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "arq"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    for noisy in ("uvicorn.access", "botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class AccessLogMiddleware:
    """One JSON line per HTTP request (replaces uvicorn's access log). Never logs headers or bodies."""

    def __init__(self, app: ASGIApp):
        self.app = app
        self.log = logging.getLogger("skf_api.access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.log.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
