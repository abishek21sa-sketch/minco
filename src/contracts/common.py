"""Shared, versioned service contracts for the MINCO platform."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CONTRACT_VERSION = "1.0"
REFERENCE_CASE = "synthetic-meridian-network"
REFERENCE_CASE_BOUNDARY = (
    "Outputs are validated only for the bundled synthetic Meridian reference case. "
    "They are not approved for real-hospital, clinical, or autonomous patient-care use."
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_correlation_id() -> str:
    return f"corr_{uuid.uuid4().hex}"


class StrictContract(BaseModel):
    """Base model used by external service contracts."""

    model_config = ConfigDict(extra="forbid", frozen=False)


class ExecutionContext(StrictContract):
    """Trace context propagated through a platform service call."""

    correlation_id: str = Field(default_factory=new_correlation_id, min_length=8, max_length=96)
    source: str = Field(default="api", min_length=1, max_length=80)
    requested_by: str | None = Field(default=None, max_length=120)


class EvidenceReference(StrictContract):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ServiceMetadata(StrictContract):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    service: str
    generated_at: datetime = Field(default_factory=utc_now)
    correlation_id: str
    reference_case: Literal["synthetic-meridian-network"] = REFERENCE_CASE
    evidence: EvidenceReference | None = None
    warnings: list[str] = Field(default_factory=list)
    claim_boundary: str = REFERENCE_CASE_BOUNDARY
