"""Typed identity and authorization posture contract."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.contracts.common import StrictContract


class IdentityResponse(StrictContract):
    """Redacted caller identity and effective role for operational traceability."""

    schema_version: Literal["1.0"] = "1.0"
    subject: str = Field(min_length=2, max_length=120)
    role: Literal["viewer", "analyst", "reviewer", "admin"]
    auth_method: Literal["none", "api_key", "bearer"]
    authenticated: bool
    authorization_enforced: bool
    permissions: list[str]
    role_hierarchy: dict[str, int]
    review_route_requires_role: Literal["reviewer"] = "reviewer"
    correlation_id: str = Field(min_length=8, max_length=96)
    autonomous_execution_permitted: Literal[False] = False
