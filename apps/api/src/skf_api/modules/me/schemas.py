from __future__ import annotations

from pydantic import BaseModel


class Me(BaseModel):
    id: str
    email: str
    name: str
    role: str
    permissions: list[str]
