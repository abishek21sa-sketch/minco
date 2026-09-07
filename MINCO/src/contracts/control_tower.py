"""Enterprise control-tower response contract.

The control tower is an aggregation boundary for operator-facing state.  It
does not create a second optimizer; it makes the existing state, FLOW-CVaR
decision, evidence chain, diagnostics, and human gate available as one
versioned payload.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from src.contracts.common import REFERENCE_CASE, StrictContract


class ControlTowerResponse(StrictContract):
    """One auditable snapshot for the executive and operations surfaces."""

    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: str = Field(min_length=8, max_length=96)
    generated_at: datetime
    correlation_id: str = Field(min_length=8, max_length=96)
    reference_case: Literal["synthetic-meridian-network"] = REFERENCE_CASE
    operating_mode: Literal["SYNTHETIC_REFERENCE_CASE"] = "SYNTHETIC_REFERENCE_CASE"
    overall_alert_level: Literal["RED", "YELLOW", "GREEN"]
    governance_state: Literal["HUMAN_REVIEW", "BLOCKED"]
    governance: dict[str, Any]
    network: dict[str, Any]
    patient_flow: dict[str, Any]
    risk: dict[str, Any]
    decision: dict[str, Any]
    diagnostics: dict[str, Any]
    evidence: dict[str, Any]
    operational_history: dict[str, Any]
    scenario_catalog: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
