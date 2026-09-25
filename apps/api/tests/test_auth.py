"""JWT verification against the (local) Better Auth JWKS: EdDSA, issuer, audience, expiry, roles."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from api_fakes import Keys

from skf_api.core.auth import TokenVerifier
from skf_api.core.errors import Unauthorized
from skf_api.core.permissions import PermissionTable
from skf_api.settings import REPO_ROOT


async def me(client: httpx.AsyncClient, token: str | None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return await client.get("/api/v1/me", headers=headers)


async def test_valid_token_returns_principal_and_permissions(client: httpx.AsyncClient, keys: Keys) -> None:
    response = await me(client, keys.token("operator"))
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "user-operator" and body["role"] == "operator"
    assert body["permissions"] == ["compute:read", "run:create_preset", "run:read", "skill:read"]


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "someone-else"},
        {"aud": "another-api"},
        {"exp": int(time.time()) - 60},
    ],
)
async def test_rejects_wrong_claims(client: httpx.AsyncClient, keys: Keys, claims: dict[str, object]) -> None:
    response = await me(client, keys.token("admin", **claims))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_rejects_bad_signature(client: httpx.AsyncClient, keys: Keys) -> None:
    assert (await me(client, keys.token("admin", key=keys.stranger))).status_code == 401


async def test_rejects_other_algorithms(client: httpx.AsyncClient, keys: Keys) -> None:
    import jwt

    forged = jwt.encode(
        {
            "sub": "x",
            "role": "admin",
            "iss": "skf-skill-studio",
            "aud": "skf-api",
            "exp": int(time.time()) + 60,
        },
        "shared-secret-" * 4,
        algorithm="HS256",
        headers={"kid": "test-key-1"},
    )
    assert (await me(client, forged)).status_code == 401


async def test_rejects_missing_or_malformed(client: httpx.AsyncClient) -> None:
    assert (await me(client, None)).status_code == 401
    assert (await me(client, "not-a-jwt")).status_code == 401
    response = await client.get("/api/v1/me", headers={"Authorization": "Basic abc"})
    assert response.status_code == 401


async def test_unknown_role_is_forbidden(client: httpx.AsyncClient, keys: Keys) -> None:
    response = await me(client, keys.token("superuser"))
    assert response.status_code == 403
    assert response.json()["error"] == {
        "code": "forbidden",
        "message": "Unknown role 'superuser'",
        "details": {},
    }


async def test_missing_role_gets_the_default(client: httpx.AsyncClient, keys: Keys) -> None:
    response = await me(client, keys.token(None, sub="new-user"))
    assert response.json()["role"] == "viewer"


async def test_key_removed_from_the_jwks_stops_verifying(keys: Keys) -> None:
    """A rotated-out (e.g. leaked) signing key must not stay trusted once the cached key set expires."""
    published: dict[str, Any] = {"jwks": keys.jwks}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps(published["jwks"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        verifier = TokenVerifier.from_jwks_url(
            f"http://127.0.0.1:{server.server_address[1]}/jwks",
            issuer="skf-skill-studio",
            audience="skf-api",
            permissions=PermissionTable.load(REPO_ROOT / "shared" / "permissions.json"),
            cache_seconds=0.2,
        )
        token = keys.token("admin")
        assert (await verifier.verify(token)).role == "admin"

        replacement = Keys()
        published["jwks"] = {"keys": [{**replacement.jwks["keys"][0], "kid": "test-key-2"}]}
        await asyncio.sleep(0.3)
        with pytest.raises(Unauthorized):
            await verifier.verify(token)
    finally:
        server.shutdown()
