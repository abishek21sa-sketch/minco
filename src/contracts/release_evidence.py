"""Typed release-provenance evidence for enterprise handoff."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from src.contracts.common import REFERENCE_CASE_BOUNDARY, StrictContract


class ReleaseEvidenceArtifact(StrictContract):
    """One bounded, hash-addressed artifact included in the release attestation."""

    path: str = Field(min_length=1, max_length=240)
    required: bool = True
    exists: bool
    status: Literal["VERIFIED", "MISSING", "UNHASHABLE"]
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)


class ReleaseEvidenceResponse(StrictContract):
    """Integrity attestation for the candidate package, not production approval."""

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    correlation_id: str = Field(min_length=8, max_length=96)
    release_id: str = Field(min_length=1, max_length=120)
    artifact_scope: Literal["MINCO_PRODUCT_V1"] = "MINCO_PRODUCT_V1"
    status: Literal["VERIFIED", "BLOCKED"]
    build_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    artifacts: list[ReleaseEvidenceArtifact]
    gates: dict[str, bool]
    missing_artifacts: list[str]
    integrity_statement: str
    claim_boundary: str = REFERENCE_CASE_BOUNDARY
    human_review_required: Literal[True] = True
    autonomous_execution_permitted: Literal[False] = False
