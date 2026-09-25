"""Bearer JWT verification against Better Auth's JWKS (EdDSA / Ed25519)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import jwt

from skf_api.core.errors import Forbidden, Unauthorized
from skf_api.core.permissions import PermissionTable

# How long a fetched JWKS is trusted. A key removed from Better Auth's JWKS stops verifying within this time,
# so there is deliberately no per-key cache (PyJWT's `cache_keys` never expires a key it has seen).
JWKS_CACHE_SECONDS = 300


@dataclass(frozen=True)
class Principal:
    id: str
    email: str
    name: str
    role: str
    permissions: frozenset[str]

    def can(self, permission: str) -> bool:
        return permission in self.permissions


class SigningKeySource(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any: ...


class TokenVerifier:
    def __init__(self, keys: SigningKeySource, *, issuer: str, audience: str, permissions: PermissionTable):
        self._keys = keys
        self._issuer = issuer
        self._audience = audience
        self._permissions = permissions

    @classmethod
    def from_jwks_url(
        cls,
        url: str,
        *,
        issuer: str,
        audience: str,
        permissions: PermissionTable,
        cache_seconds: float = JWKS_CACHE_SECONDS,
    ) -> TokenVerifier:
        client = jwt.PyJWKClient(url, lifespan=cache_seconds, timeout=5)
        return cls(client, issuer=issuer, audience=audience, permissions=permissions)

    async def verify(self, token: str) -> Principal:
        try:
            # PyJWKClient fetches (and caches) the key set with blocking urllib.
            signing_key = await asyncio.to_thread(self._keys.get_signing_key_from_jwt, token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["EdDSA"],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": ["exp", "sub", "iss", "aud"]},
            )
        except jwt.PyJWKClientError as exc:
            raise Unauthorized("Token signing key not found") from exc
        except jwt.InvalidTokenError as exc:
            raise Unauthorized(f"Invalid token: {exc}") from exc

        role = claims.get("role") or self._permissions.default_role
        permissions = self._permissions.for_role(role)
        if permissions is None:
            raise Forbidden(f"Unknown role '{role}'")
        return Principal(
            id=str(claims["sub"]),
            email=str(claims.get("email", "")),
            name=str(claims.get("name") or claims.get("email") or claims["sub"]),
            role=role,
            permissions=permissions,
        )


def bearer_token(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise Unauthorized("Missing bearer token")
    return token.strip()
