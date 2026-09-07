"""Typed operational-event history and replay-state contract."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from src.contracts.common import REFERENCE_CASE, REFERENCE_CASE_BOUNDARY, StrictContract


class OperationalHistoryResponse(StrictContract):
    """A traceable event-lake/replay snapshot, not a live-feed authorization."""

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    correlation_id: str = Field(min_length=8, max_length=96)
    reference_case: Literal["synthetic-meridian-network"] = REFERENCE_CASE
    data_mode: Literal[
        "OPERATIONAL_EVENT_LAKE",
        "SYNTHETIC_HISTORICAL_REPLAY",
        "UNAVAILABLE",
    ]
    data_stack: str
    event_count: int = Field(ge=0)
    latest_event_timestamp: datetime | None = None
    latest_ingested_at: datetime | None = None
    as_of: datetime
    freshness_status: Literal["REPLAY_ONLY", "CURRENT", "STALE", "UNAVAILABLE"]
    freshness: dict[str, Any]
    event_type_counts: dict[str, int]
    source_modes: list[str]
    hospital_ids: list[str]
    reconstructed_state: dict[str, Any]
    evidence: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    live_feed_connected: bool = False
    autonomous_execution_permitted: Literal[False] = False
    claim_boundary: str = REFERENCE_CASE_BOUNDARY
