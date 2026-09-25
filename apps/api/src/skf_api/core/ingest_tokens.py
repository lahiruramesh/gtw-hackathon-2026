"""Per-stage ingest tokens: `base64url(stage_id).exp_unix.hex(hmac_sha256(key, "stage_id.exp"))`.

Stateless (no DB lookup to verify the signature); the ingest router additionally checks that the stage
exists and is not terminal, so a token dies with its stage even before it expires.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass

from skf_api.core.crypto import derive_key


class InvalidIngestToken(Exception):
    pass


@dataclass(frozen=True)
class IngestClaims:
    stage_id: uuid.UUID
    expires_at: int


def _b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _unb64(value: str) -> str:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()


class IngestTokenSigner:
    def __init__(self, secret_key: str):
        self._key = derive_key(secret_key, "ingest-tokens")

    def _sign(self, stage_id: str, exp: int) -> str:
        return hmac.new(self._key, f"{stage_id}.{exp}".encode(), hashlib.sha256).hexdigest()

    def mint(self, stage_id: uuid.UUID, expires_at: int) -> str:
        sid = str(stage_id)
        return f"{_b64(sid)}.{expires_at}.{self._sign(sid, expires_at)}"

    def verify(self, token: str, *, now: float | None = None) -> IngestClaims:
        try:
            encoded_sid, exp_text, signature = token.split(".")
            sid = _unb64(encoded_sid)
            exp = int(exp_text)
            stage_id = uuid.UUID(sid)
        except ValueError as exc:
            raise InvalidIngestToken("malformed token") from exc
        if not hmac.compare_digest(signature, self._sign(sid, exp)):
            raise InvalidIngestToken("bad signature")
        if exp < (time.time() if now is None else now):
            raise InvalidIngestToken("token expired")
        return IngestClaims(stage_id=stage_id, expires_at=exp)
