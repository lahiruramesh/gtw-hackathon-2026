"""Keys derived from SECRET_KEY: Fernet for compute-target secrets, HMAC key for ingest tokens."""

from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def derive_key(secret_key: str, purpose: str) -> bytes:
    """32-byte key for one purpose, so rotating or leaking one use never exposes another."""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"skf-skill-studio", info=purpose.encode())
    return hkdf.derive(secret_key.encode())


class SecretDecryptError(Exception):
    pass


class SecretBox:
    """Encrypts JSON secrets at rest. Plaintext never leaves the worker/API process."""

    def __init__(self, secret_key: str):
        self._fernet = Fernet(base64.urlsafe_b64encode(derive_key(secret_key, "compute-target-secrets")))

    def encrypt_json(self, value: dict[str, Any]) -> bytes:
        return self._fernet.encrypt(json.dumps(value, separators=(",", ":")).encode())

    def decrypt_json(self, token: bytes) -> dict[str, Any]:
        try:
            return json.loads(self._fernet.decrypt(token))
        except InvalidToken as exc:
            raise SecretDecryptError("stored secret cannot be decrypted with the current SECRET_KEY") from exc
