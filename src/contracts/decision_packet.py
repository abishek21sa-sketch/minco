"""Evidence-backed decision-packet contract for governed operator review."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from src.contracts.common import REFERENCE_CASE_BOUNDARY, StrictContract, utc_now
from src.contracts.release_evidence import ReleaseEvidenceResponse
from src.contracts.review import AuditIntegrityResponse, DecisionReviewResponse


PacketReviewStatus = Literal[
    "PENDING",
    "ACCEPTED_FOR_OPERATIONS_REVIEW",
    "DEFERRED",
    "REJECTED",
]


class DecisionPacketGovernance(StrictContract):
    """Explicit authorization boundary carried with every review packet."""

    packet_ready: bool
    review_status: PacketReviewStatus
    human_review_required: Literal[True] = True
    autonomous_execution_permitted: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)
    claim_boundary: str = REFERENCE_CASE_BOUNDARY


class DecisionPacketResponse(StrictContract):
    """A portable, auditable bundle for one governed decision run."""

    schema_version: Literal["1.0"] = "1.0"
    packet_id: str = Field(min_length=12, max_length=140)
    generated_at: datetime = Field(default_factory=utc_now)
    correlation_id: str = Field(min_length=8, max_length=96)
    run_id: str = Field(min_length=8, max_length=96)
    decision_run: dict[str, Any]
    what_if_run: dict[str, Any] | None = None
    manifest_reference: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    reviews: list[DecisionReviewResponse] = Field(default_factory=list)
    audit_integrity: AuditIntegrityResponse
    release_evidence: ReleaseEvidenceResponse
    governance: DecisionPacketGovernance
    autonomous_execution_permitted: Literal[False] = False
