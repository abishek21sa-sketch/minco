"""Versioned contracts for the governed live-event ingestion boundary."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field, field_validator

from src.contracts.common import REFERENCE_CASE_BOUNDARY, StrictContract, utc_now
from src.hospital_events.models import HospitalEvent


class EventIngestionRequest(StrictContract):
    """A bounded batch from an approved external operational stream."""

    schema_version: Literal["1.0"] = "1.0"
    source_system: str = Field(min_length=1, max_length=120)
    source_mode: Literal["external_stream"] = "external_stream"
    received_at: datetime = Field(default_factory=utc_now)
    events: list[HospitalEvent] = Field(min_length=1, max_length=500)

    @field_validator("received_at")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware")
        return value


class EventIngestionResponse(StrictContract):
    """Auditable result of validating and appending one external event batch."""

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    correlation_id: str = Field(min_length=8, max_length=96)
    source_system: str
    source_mode: Literal["external_stream"] = "external_stream"
    status: Literal["ACCEPTED", "PARTIAL", "REJECTED", "UNAVAILABLE"]
    accepted_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    accepted_event_ids: list[str] = Field(default_factory=list)
    rejected_events: list[dict[str, Any]] = Field(default_factory=list)
    latest_event_timestamp: datetime | None = None
    freshness_status: Literal["CURRENT", "STALE", "UNAVAILABLE"]
    live_feed_connected: bool = False
    validation_policy: dict[str, Any]
    claim_boundary: str = REFERENCE_CASE_BOUNDARY
    autonomous_execution_permitted: Literal[False] = False


class EventIngestionStatusResponse(StrictContract):
    """Current gateway posture, deliberately separate from production approval."""

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    correlation_id: str = Field(min_length=8, max_length=96)
    gateway_available: bool
    live_feed_connected: bool
    data_mode: Literal["DISCONNECTED", "EXTERNAL_STREAM"]
    source_systems: list[str]
    accepted_event_count: int = Field(ge=0)
    latest_event_timestamp: datetime | None = None
    latest_ingested_at: datetime | None = None
    freshness_status: Literal["CURRENT", "STALE", "UNAVAILABLE"]
    validation_policy: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    claim_boundary: str = REFERENCE_CASE_BOUNDARY
    autonomous_execution_permitted: Literal[False] = False
