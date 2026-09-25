"""Keyset (cursor) pagination: `{"items": [...], "next_cursor": str | null}` with `?limit=&cursor=`."""

from __future__ import annotations

import base64
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any

import orjson
import sqlalchemy as sa
from fastapi import Query
from pydantic import BaseModel
from sqlalchemy.orm import QueryableAttribute

from skf_api.core.errors import Invalid

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None


class PageParams(BaseModel):
    limit: int = DEFAULT_LIMIT
    cursor: str | None = None


def page_params(
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    cursor: Annotated[str | None, Query()] = None,
) -> PageParams:
    return PageParams(limit=limit, cursor=cursor)


def _encode_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _decode_value(column: QueryableAttribute[Any], raw: Any) -> Any:
    python_type = column.type.python_type
    if python_type is datetime:
        return datetime.fromisoformat(raw)
    if python_type is uuid.UUID:
        return uuid.UUID(raw)
    return python_type(raw)


class Keyset:
    """Orders by `columns` (all descending or all ascending) and continues after the last row's values."""

    def __init__(self, *columns: QueryableAttribute[Any], descending: bool = True):
        self.columns = columns
        self.descending = descending

    def apply(self, stmt: sa.Select[Any], params: PageParams) -> sa.Select[Any]:
        order = [c.desc() if self.descending else c.asc() for c in self.columns]
        stmt = stmt.order_by(*order).limit(params.limit + 1)
        if params.cursor:
            values = self._decode(params.cursor)
            key = sa.tuple_(*self.columns)
            stmt = stmt.where(key < sa.tuple_(*values) if self.descending else key > sa.tuple_(*values))
        return stmt

    def next_cursor(self, rows: Sequence[Any], params: PageParams, key_of: Any) -> str | None:
        """`rows` is what `apply` fetched (limit + 1); `key_of(row)` returns the row's key values."""
        if len(rows) <= params.limit:
            return None
        last = rows[params.limit - 1]
        raw = orjson.dumps([_encode_value(v) for v in key_of(last)])
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def _decode(self, cursor: str) -> list[Any]:
        try:
            raw = orjson.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
            if not isinstance(raw, list) or len(raw) != len(self.columns):
                raise ValueError("cursor shape")
            return [_decode_value(c, v) for c, v in zip(self.columns, raw, strict=True)]
        except (ValueError, TypeError, orjson.JSONDecodeError) as exc:
            raise Invalid("Invalid cursor") from exc
