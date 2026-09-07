"""Operational-state contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from src.contracts.common import ServiceMetadata, StrictContract


class OperationalStateSnapshot(StrictContract):
    snapshot_id: str
    observed_at: datetime
    state_source: Literal["synthetic_reference_case"] = "synthetic_reference_case"
    hospital_count: int = Field(ge=1)
    hospital_ids: list[str]
    cohort_count: int = Field(ge=1)
    cohort_ids: list[str]
    horizon_days: int = Field(ge=1)
    arrival_rows: int = Field(ge=1)
    total_expected_arrivals: float = Field(ge=0.0)
    capacity_by_resource: dict[str, float]
    safe_utilization_min: float = Field(gt=0.0, lt=1.0)
    safe_utilization_max: float = Field(gt=0.0, lt=1.0)
    allowed_transfer_lanes: int = Field(ge=0)
    total_transfer_capacity_per_day: float = Field(ge=0.0)
    input_fingerprints: dict[str, str]


class OperationalStateResponse(StrictContract):
    metadata: ServiceMetadata
    state: OperationalStateSnapshot
