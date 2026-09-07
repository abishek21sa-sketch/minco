"""Decision-orchestration contracts."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.contracts.common import ExecutionContext, ServiceMetadata, StrictContract
from src.contracts.metrics import OperationalMetrics
from src.contracts.scenario import ScenarioDefinition, ScenarioParameters


class DecisionRecommendationRequest(StrictContract):
    scenario_id: str = Field(default="baseline", min_length=2, max_length=64)
    overrides: ScenarioParameters | None = None
    scenario_label: str | None = Field(default=None, max_length=160)
    n_replications: int = Field(default=50, ge=1, le=200)
    context: ExecutionContext = Field(default_factory=ExecutionContext)


class AlertContract(StrictContract):
    level: Literal["RED", "YELLOW", "GREEN"]
    trigger: str
    message: str
    timestamp: str


class RecommendationAction(StrictContract):
    action_id: str
    title: str
    priority: Literal["immediate", "monitor", "routine"]
    rationale: str
    expected_effects: list[str]
    assumptions: list[str]
    tradeoffs: list[str]
    human_review_required: bool = True


class DecisionRecommendationResponse(StrictContract):
    run_id: str
    overall_alert_level: Literal["RED", "YELLOW", "GREEN"]
    scenario: ScenarioDefinition
    metrics: OperationalMetrics
    alerts: list[AlertContract]
    recommendation: RecommendationAction
    executive_brief: str
    metadata: ServiceMetadata
