"""Typed system health and operational-readiness contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.contracts.common import REFERENCE_CASE, StrictContract


class HealthResponse(StrictContract):
    """Liveness response that does not depend on optional optimization software."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok"] = "ok"
    service: str
    version: str
    contract_version: Literal["1.0"] = "1.0"
    reference_case: Literal["synthetic-meridian-network"] = REFERENCE_CASE
    correlation_id: str = Field(min_length=8, max_length=96)


class ReadinessResponse(StrictContract):
    """Dependency/readiness response with explicit non-authorization semantics."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ready", "degraded", "not_ready"]
    service: str
    version: str
    contract_version: Literal["1.0"] = "1.0"
    reference_case: Literal["synthetic-meridian-network"] = REFERENCE_CASE
    correlation_id: str = Field(min_length=8, max_length=96)
    data_ready: bool
    solver_package_available: bool
    solver_license_verified: bool
    solver_status: str
    solver_version: str | None = None
    solver_check_seconds: float = Field(ge=0.0)
    checks: dict[str, bool]
    note: str
    release_readiness: Literal["NOT_FOR_PRODUCTION"] = "NOT_FOR_PRODUCTION"
    autonomous_execution_permitted: Literal[False] = False
