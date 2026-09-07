"""Typed release-readiness scorecard for MINCO enterprise operation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from src.contracts.common import REFERENCE_CASE, REFERENCE_CASE_BOUNDARY, StrictContract


class ReleaseReadinessResponse(StrictContract):
    """Evidence-backed deployment posture with explicit blockers and boundaries."""

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    correlation_id: str = Field(min_length=8, max_length=96)
    reference_case: Literal["synthetic-meridian-network"] = REFERENCE_CASE
    status: Literal["CONDITIONAL", "BLOCKED"]
    release_readiness: Literal["NOT_FOR_PRODUCTION"] = "NOT_FOR_PRODUCTION"
    checks: dict[str, bool]
    dependency_status: dict[str, Any]
    security_controls: dict[str, Any]
    evidence_artifacts: dict[str, Any]
    blocking_reasons: list[str]
    next_actions: list[str]
    human_review_required: Literal[True] = True
    autonomous_execution_permitted: Literal[False] = False
    claim_boundary: str = REFERENCE_CASE_BOUNDARY
