from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RuntimeStatus(BaseModel):
    product: str = "MINCO Native Hospital Operations Workstation"
    version: str
    runtime_contract_version: str = "2.0"
    data_mode: str
    data_stack: str
    event_count: int
    latest_event_timestamp: datetime | None
    primary_or_solver: str = "Julia/JuMP/Gurobi"
    solver_license_mode: str = "academic"
    solver_fallback_enabled: bool = True
    facility_count: int = 3
    operating_mode: str = "normal"
    available_operating_modes: list[str] = Field(default_factory=list)
    dataset_provenance: dict[str, object] = Field(default_factory=dict)
    monte_carlo_engine: str = "Python/NumPy common-random-number laboratory"
    claude_provider: str = "Anthropic Claude"
    evidence_boundary: str
    warnings: list[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    planning_horizon_periods: int = Field(default=4, ge=1, le=12)
    n_scenarios: int = Field(default=30, ge=2, le=500)
    risk_alpha: float = Field(default=0.95, gt=0.5, lt=1.0)
    risk_weight: float = Field(default=0.40, ge=0.0, le=5.0)
    risk_posture: Literal["balanced", "service_first", "cost_first"] = "balanced"
    use_primary_julia_solver: bool = False
    operating_mode: str = "normal"
    # Keep the legacy in-process API oracle fast by default. The native
    # enterprise workstation explicitly requests 12+ facilities, while the
    # replay/event lake remains 120-facility by default.
    network_hospitals: int = Field(default=3, ge=3, le=50)


class InterventionRequest(BaseModel):
    extra_ward_beds: int = Field(default=0, ge=0, le=100)
    extra_icu_beds: int = Field(default=0, ge=0, le=100)
    extra_ed_servers: int = Field(default=0, ge=0, le=50)
    transfer_out_capacity_per_6h: int = Field(default=0, ge=0, le=100)
    n_replications: int = Field(default=120, ge=10, le=2000)
    operating_mode: str = "normal"


class ReviewRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    evidence_scope: list[str] = Field(default_factory=lambda: ["state", "plan", "simulation"])
    execute_external_llm: bool = False
