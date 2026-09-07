"""Contracts for MINCO's integrated surge-response planning workflow."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.contracts.common import ExecutionContext, ServiceMetadata, StrictContract
from src.contracts.scenario import ScenarioEvaluationResponse, ScenarioParameters


class SurgePlanRequest(StrictContract):
    scenario_id: str = Field(default="baseline", min_length=2, max_length=64)
    overrides: ScenarioParameters | None = None
    n_replications: int = Field(default=20, ge=1, le=200)
    planning_quantile: Literal["p50", "p95"] = "p95"
    context: ExecutionContext = Field(default_factory=ExecutionContext)


class DemandForecastItem(StrictContract):
    forecast_date: str
    hospital_id: str
    cohort: str
    predicted_p05: float = Field(ge=0.0)
    predicted_p50: float = Field(ge=0.0)
    predicted_p95: float = Field(ge=0.0)
    seasonal_naive: float = Field(ge=0.0)
    evidence_label: Literal["PREDICTED"] = "PREDICTED"
    validation_scope: Literal["synthetic_reference_case_only"] = "synthetic_reference_case_only"


class DemandForecastSummary(StrictContract):
    forecast_date: str
    network_p05: float = Field(ge=0.0)
    network_p50: float = Field(ge=0.0)
    network_p95: float = Field(ge=0.0)
    seasonal_naive_total: float = Field(gt=0.0)
    p95_to_seasonal_naive_ratio: float = Field(gt=0.0)
    planning_quantile: Literal["p50", "p95"]
    planning_to_seasonal_naive_ratio: float = Field(gt=0.0)
    scenario_demand_multiplier: float = Field(gt=0.0)
    effective_optimizer_demand_multiplier: float = Field(gt=0.0, le=5.0)
    evidence_label: Literal["PREDICTED"] = "PREDICTED"


class ResourcePressureItem(StrictContract):
    hospital_id: str
    resource: str
    source_quantile: str
    expected_census: float = Field(ge=0.0)
    base_capacity: float = Field(gt=0.0)
    safe_utilization: float = Field(gt=0.0, lt=1.0)
    safe_capacity: float = Field(gt=0.0)
    projected_utilization: float = Field(ge=0.0)
    safe_capacity_gap: float
    above_safe_capacity: bool
    calculation_label: Literal["CALCULATED"] = "CALCULATED"


class OptimizedActionSummary(StrictContract):
    solver_status: str | None = None
    objective_value: float | None = None
    total_surge_activated: float = Field(default=0.0, ge=0.0)
    total_elective_rejected: float = Field(default=0.0, ge=0.0)
    total_icu_transfer_load: float = Field(default=0.0, ge=0.0)
    evidence_label: Literal["OPTIMIZED"] = "OPTIMIZED"


class SurgeDecision(StrictContract):
    action_id: str
    title: str
    priority: Literal["immediate", "monitor", "routine"]
    rationale: list[str]
    human_review_required: bool = True
    evidence_label: Literal["RECOMMENDED_ACTION"] = "RECOMMENDED_ACTION"


class SurgePlanResponse(StrictContract):
    run_id: str
    forecast_summary: DemandForecastSummary
    forecasts: list[DemandForecastItem]
    resource_pressure_p50: list[ResourcePressureItem]
    resource_pressure_p95: list[ResourcePressureItem]
    scenario_evaluation: ScenarioEvaluationResponse
    optimized_actions: OptimizedActionSummary
    decision: SurgeDecision
    metadata: ServiceMetadata
