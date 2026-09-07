"""Deployment-boundary authentication and role-based authorization for MINCO."""
from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import Request, Response
from starlette.responses import JSONResponse


API_KEY_HEADER = "X-API-Key"
AUTHORIZATION_HEADER = "Authorization"
PUBLIC_PATHS = frozenset({"/", "/health", "/readiness", "/docs", "/openapi.json", "/redoc"})
ROLE_LEVELS = {"viewer": 10, "analyst": 20, "reviewer": 30, "admin": 40}
ROLE_PERMISSIONS = {
    "viewer": ("read:operations", "read:evidence", "read:audit"),
    "analyst": ("read:operations", "read:evidence", "read:audit", "evaluate:scenarios", "ingest:events"),
    "reviewer": ("read:operations", "read:evidence", "read:audit", "evaluate:scenarios", "ingest:events", "review:decisions"),
    "admin": ("read:operations", "read:evidence", "read:audit", "evaluate:scenarios", "ingest:events", "review:decisions", "admin:configure"),
}
AuthMethod = Literal["none", "api_key", "bearer"]


@dataclass(frozen=True)
class IdentityContext:
    """Normalized identity carried on the request without retaining credentials."""

    subject: str
    role: str
    auth_method: AuthMethod
    authenticated: bool

    @property
    def permissions(self) -> tuple[str, ...]:
        return ROLE_PERMISSIONS.get(self.role, ROLE_PERMISSIONS["viewer"])

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "role": self.role,
            "auth_method": self.auth_method,
            "authenticated": self.authenticated,
            "permissions": list(self.permissions),
        }


def _normalize_role(value: object) -> str:
    role = str(value or "viewer").strip().lower()
    return role if role in ROLE_LEVELS else "viewer"


def configured_api_key() -> str:
    """Return the legacy deployment API key, if one is configured."""
    return os.getenv("MINCO_API_KEY", "").strip()


def configured_credentials() -> list[tuple[str, IdentityContext]]:
    """Return configured credentials and redacted identity metadata only."""
    credentials: list[tuple[str, IdentityContext]] = []
    raw = os.getenv("MINCO_API_KEYS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, dict):
            for index, (credential, metadata) in enumerate(parsed.items(), start=1):
                if not isinstance(credential, str) or not credential.strip():
                    continue
                if isinstance(metadata, dict):
                    role = _normalize_role(metadata.get("role"))
                    subject = str(metadata.get("subject") or f"configured-operator-{index}")
                else:
                    role = _normalize_role(metadata)
                    subject = f"configured-operator-{index}"
                credentials.append(
                    (
                        credential.strip(),
                        IdentityContext(subject=subject[:120], role=role, auth_method="api_key", authenticated=True),
                    )
                )

    legacy = configured_api_key()
    if legacy:
        credentials.append(
            (
                legacy,
                IdentityContext(
                    subject=os.getenv("MINCO_API_KEY_SUBJECT", "legacy-api-key-operator")[:120],
                    role=_normalize_role(os.getenv("MINCO_API_KEY_ROLE", "admin")),
                    auth_method="api_key",
                    authenticated=True,
                ),
            )
        )
    return credentials


def api_key_enforcement_enabled() -> bool:
    """Whether deployment credentials will be required for protected routes."""
    return bool(configured_credentials())


def _supplied_token(request: Request) -> tuple[str, AuthMethod]:
    api_key = request.headers.get(API_KEY_HEADER, "").strip()
    if api_key:
        return api_key, "api_key"
    authorization = request.headers.get(AUTHORIZATION_HEADER, "").strip()
    if re.match(r"^Bearer\s+", authorization, flags=re.IGNORECASE):
        return re.sub(r"^Bearer\s+", "", authorization, flags=re.IGNORECASE).strip(), "bearer"
    return "", "api_key"


def identity_for_request(request: Request) -> IdentityContext:
    """Return the normalized identity assigned by middleware."""
    return getattr(
        request.state,
        "identity",
        IdentityContext(
            subject="local-reference-operator",
            role="admin",
            auth_method="none",
            authenticated=False,
        ),
    )


def required_role(method: str, path: str) -> str:
    """Map route intent to the minimum enterprise role."""
    if method.upper() == "GET":
        return "viewer"
    if re.fullmatch(r"/v1/audit/[^/]+/review", path):
        return "reviewer"
    if path.endswith("/operational-events/ingest"):
        return "analyst"
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        return "analyst"
    return "viewer"


def _match_identity(request: Request) -> IdentityContext | None:
    supplied, supplied_method = _supplied_token(request)
    if not supplied:
        return None
    for expected, identity in configured_credentials():
        if secrets.compare_digest(supplied, expected):
            return IdentityContext(
                subject=identity.subject,
                role=identity.role,
                auth_method=supplied_method,
                authenticated=True,
            )
    return None


def _authentication_error() -> JSONResponse:
    response = JSONResponse(
        status_code=401,
        content={
            "detail": {
                "code": "authentication_required",
                "message": "A valid X-API-Key or Authorization: Bearer credential is required for operational routes.",
            }
        },
    )
    response.headers["WWW-Authenticate"] = "ApiKey, Bearer"
    return response


async def api_key_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Authenticate and enforce the minimum role for deployment-protected routes."""
    identity = _match_identity(request)
    if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
        request.state.identity = identity or IdentityContext(
            subject="anonymous-probe",
            role="viewer",
            auth_method="none",
            authenticated=False,
        )
        return await call_next(request)

    if not api_key_enforcement_enabled():
        request.state.identity = IdentityContext(
            subject="local-reference-operator",
            role="admin",
            auth_method="none",
            authenticated=False,
        )
        return await call_next(request)

    if identity is None:
        return _authentication_error()

    request.state.identity = identity
    minimum = required_role(request.method, request.url.path)
    if ROLE_LEVELS.get(identity.role, 0) < ROLE_LEVELS[minimum]:
        return JSONResponse(
            status_code=403,
            content={
                "detail": {
                    "code": "insufficient_role",
                    "message": f"Role '{minimum}' is required for this operation.",
                    "required_role": minimum,
                    "subject": identity.subject,
                    "role": identity.role,
                }
            },
        )
    return await call_next(request)


__all__ = [
    "API_KEY_HEADER",
    "AUTHORIZATION_HEADER",
    "PUBLIC_PATHS",
    "ROLE_LEVELS",
    "ROLE_PERMISSIONS",
    "IdentityContext",
    "api_key_enforcement_enabled",
    "api_key_middleware",
    "configured_api_key",
    "configured_credentials",
    "identity_for_request",
    "required_role",
]
