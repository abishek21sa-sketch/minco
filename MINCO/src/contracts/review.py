"""Human-review contracts for governed operational recommendations."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from src.contracts.common import StrictContract, utc_now


REVIEW_DECISIONS = (
    "ACCEPTED_FOR_OPERATIONS_REVIEW",
    "DEFERRED",
    "REJECTED",
)
ReviewDecision = Literal[
    "ACCEPTED_FOR_OPERATIONS_REVIEW",
    "DEFERRED",
    "REJECTED",
]


class DecisionReviewRequest(StrictContract):
    """A human disposition of a persisted recommendation run."""

    reviewed_by: str = Field(min_length=2, max_length=120)
    decision: ReviewDecision
    comment: str = Field(min_length=1, max_length=2000)


class DecisionReviewResponse(StrictContract):
    """Immutable review event returned after a human disposition is recorded."""

    schema_version: Literal["1.0"] = "1.0"
    review_id: str = Field(min_length=8, max_length=96)
    run_id: str = Field(min_length=8, max_length=96)
    reviewed_at: datetime = Field(default_factory=utc_now)
    reviewed_by: str = Field(min_length=2, max_length=120)
    decision: ReviewDecision
    comment: str = Field(min_length=1, max_length=2000)
    hash_algorithm: Literal["SHA-256"] = "SHA-256"
    previous_hash: str = Field(pattern=r"^(GENESIS|[0-9a-f]{64})$")
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    autonomous_execution_permitted: Literal[False] = False
    governance_note: str = (
        "This review records human disposition only. It never dispatches staffing, transfer, "
        "diversion, or clinical actions."
    )


class AuditIntegrityResponse(StrictContract):
    """Verification result for the append-only human-review hash chain."""

    schema_version: Literal["1.0"] = "1.0"
    checked_at: datetime
    status: Literal["VALID", "INVALID"]
    review_count: int = Field(ge=0)
    hash_algorithm: Literal["SHA-256"] = "SHA-256"
    genesis_hash: Literal["GENESIS"] = "GENESIS"
    head_hash: str = Field(pattern=r"^(GENESIS|[0-9a-f]{64})$")
    violations: list[dict[str, str]] = Field(default_factory=list)
    autonomous_execution_permitted: Literal[False] = False
