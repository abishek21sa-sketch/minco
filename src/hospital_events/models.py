from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class EventType(StrEnum):
    ARRIVAL = "arrival"
    ADMISSION = "admission"
    BED_REQUEST = "bed_request"
    BED_ASSIGNED = "bed_assigned"
    TRANSFER_REQUEST = "transfer_request"
    TRANSFER_COMPLETE = "transfer_complete"
    DISCHARGE_READY = "discharge_ready"
    DISCHARGE = "discharge"
    CAPACITY_CHANGE = "capacity_change"
    STAFFING_CHANGE = "staffing_change"
    DIVERSION = "diversion"


class SourceMode(StrEnum):
    SYNTHETIC = "synthetic"
    FILE_BATCH = "file_batch"
    HISTORICAL_REPLAY = "historical_replay"
    EXTERNAL_STREAM = "external_stream"


class HospitalEvent(BaseModel):
    """Canonical immutable event accepted by the Phase-2 operational runtime.

    The event schema deliberately separates the operational timestamp from
    ingestion time and requires source provenance. Missing required concepts are
    rejected rather than fabricated.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_timestamp: datetime
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hospital_id: str = Field(min_length=1)
    event_type: EventType
    patient_pathway_or_cohort: str = Field(min_length=1)
    resource: str = Field(min_length=1)
    quantity: float = 1.0
    source_system: str = Field(min_length=1)
    source_mode: SourceMode = SourceMode.FILE_BATCH
    unit_id: str | None = None
    patient_id: str | None = None
    from_hospital_id: str | None = None
    to_hospital_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_timestamp", "ingested_at")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @field_validator("quantity")
    @classmethod
    def _finite_quantity(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("quantity must be finite")
        return value

    @model_validator(mode="after")
    def _event_semantics(self) -> "HospitalEvent":
        if self.event_type == EventType.TRANSFER_COMPLETE:
            if not self.from_hospital_id or not self.to_hospital_id:
                raise ValueError("transfer_complete requires from_hospital_id and to_hospital_id")
            if self.from_hospital_id == self.to_hospital_id:
                raise ValueError("transfer origin and destination must differ")
        if self.event_type in {EventType.CAPACITY_CHANGE, EventType.STAFFING_CHANGE} and self.quantity == 0:
            raise ValueError("capacity/staffing change quantity cannot be zero")
        return self
