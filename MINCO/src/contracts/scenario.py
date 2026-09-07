"""Scenario-management and scenario-evaluation contracts."""
from __future__ import annotations

from pydantic import Field, model_validator

from src.contracts.common import ExecutionContext, ServiceMetadata, StrictContract
from src.contracts.metrics import OperationalMetrics


class ScenarioParameters(StrictContract):
    icu_bed_delta: int = Field(default=0, ge=-500, le=500)
    transfer_capacity_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)
    demand_surge_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)


class ScenarioDefinition(StrictContract):
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    name: str
    description: str
    parameters: ScenarioParameters
    built_in: bool = True


class ScenarioEvaluationRequest(StrictContract):
    scenario_id: str = Field(default="baseline", min_length=2, max_length=64)
    overrides: ScenarioParameters | None = None
    n_replications: int = Field(default=10, ge=1, le=200)
    context: ExecutionContext = Field(default_factory=ExecutionContext)

    @model_validator(mode="after")
    def custom_requires_parameters(self) -> "ScenarioEvaluationRequest":
        if self.scenario_id == "custom" and self.overrides is None:
            raise ValueError("custom scenario evaluation requires overrides")
        return self


class ScenarioEvaluationResponse(StrictContract):
    run_id: str
    scenario: ScenarioDefinition
    metrics: OperationalMetrics
    metadata: ServiceMetadata
