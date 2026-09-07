"""Typed operational metrics contract for platform monitoring."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from src.contracts.common import REFERENCE_CASE_BOUNDARY, StrictContract


class ObservabilityMetricsResponse(StrictContract):
    """Low-cardinality API telemetry suitable for SLO and incident review."""

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    correlation_id: str = Field(min_length=8, max_length=96)
    service: Literal["minco-api"] = "minco-api"
    uptime_seconds: float = Field(ge=0.0)
    request_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    status_counts: dict[str, int]
    route_counts: dict[str, int]
    latency_ms: dict[str, float]
    data_logging_policy: dict[str, Any]
    claim_boundary: str = REFERENCE_CASE_BOUNDARY
    autonomous_execution_permitted: Literal[False] = False
