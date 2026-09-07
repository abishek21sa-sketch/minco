"""Decision orchestration over validated analytical services."""
from __future__ import annotations

from pathlib import Path

from src.analysis.stress_classification import classify_stress_label
from src.command_center.alert_engine import evaluate_alerts, highest_alert_level
from src.command_center.executive_brief_generator import generate_executive_brief
from src.config.paths import AUDIT_DB_PATH
from src.contracts.common import EvidenceReference, ServiceMetadata
from src.contracts.decision import (
    AlertContract,
    DecisionRecommendationRequest,
    DecisionRecommendationResponse,
    RecommendationAction,
)
from src.contracts.scenario import ScenarioEvaluationRequest
from src.services.scenario_service import ScenarioService
from src.storage.audit_repository import (
    log_alerts,
    log_decision_run,
    log_recommendation,
    log_run_manifest,
)
from src.validation.run_manifest import RUN_MANIFEST_DIR, write_run_manifest


def _recommendation_for_level(level: str) -> RecommendationAction:
    if level == "RED":
        return RecommendationAction(
            action_id="activate_surge_and_coordinate_transfers",
            title="Activate surge capacity and coordinate transfers",
            priority="immediate",
            rationale=(
                "The evaluated network state breaches one or more operational pressure thresholds."
            ),
            expected_effects=[
                "Reduce unsafe excess demand",
                "Lower avoidable blocking and overflow pressure",
                "Improve network resilience through coordinated transfers",
            ],
            assumptions=[
                "Surge staffing can be made available",
                "Transfer lanes remain operational at the evaluated capacity",
                "Reference-case forecasts remain within validated synthetic bounds",
            ],
            tradeoffs=["Higher surge and transfer cost", "Additional coordination workload"],
        )
    if level == "YELLOW":
        return RecommendationAction(
            action_id="maintain_coordination_and_monitor",
            title="Maintain coordinated policy and increase monitoring",
            priority="monitor",
            rationale="The network is stressed but has not reached the critical escalation threshold.",
            expected_effects=["Preserve flow balance", "Detect deterioration before escalation"],
            assumptions=["No major unmodeled disruption occurs"],
            tradeoffs=["Continued monitoring effort", "Potentially conservative resource posture"],
        )
    return RecommendationAction(
        action_id="continue_coordinated_operation",
        title="Continue coordinated operation",
        priority="routine",
        rationale="All monitored synthetic reference-case thresholds remain within normal range.",
        expected_effects=["Maintain current service performance"],
        assumptions=["Demand and capacity remain near the evaluated state"],
        tradeoffs=["No proactive surge capacity is activated"],
    )


class DecisionOrchestrationService:
    """Transform a scenario analysis into an audited human-review recommendation."""

    service_name = "decision_orchestration_service"

    def __init__(
        self,
        scenario_service: ScenarioService,
        *,
        db_path: Path = AUDIT_DB_PATH,
        manifest_dir: Path = RUN_MANIFEST_DIR,
    ) -> None:
        self.scenario_service = scenario_service
        self.db_path = Path(db_path)
        self.manifest_dir = Path(manifest_dir)

    def recommend(
        self,
        request: DecisionRecommendationRequest,
    ) -> DecisionRecommendationResponse:
        evaluation_request = ScenarioEvaluationRequest(
            scenario_id=request.scenario_id,
            overrides=request.overrides,
            n_replications=request.n_replications,
            context=request.context,
        )
        scenario = self.scenario_service.resolve_scenario(evaluation_request)
        metrics = self.scenario_service.analyze(scenario, request.n_replications)
        state = metrics.model_dump()

        alert_objects = evaluate_alerts(state, classify_fn=classify_stress_label)
        overall_level = highest_alert_level(alert_objects)
        label = request.scenario_label or scenario.name
        brief = generate_executive_brief(state, alert_objects, scenario_label=label)
        recommendation = _recommendation_for_level(overall_level)

        run_id = log_decision_run(
            scenario=label,
            overall_alert=overall_level,
            pipeline_mode=request.context.source,
            runtime_seconds=metrics.solve_runtime_seconds,
            db_path=self.db_path,
        )
        log_recommendation(
            run_id=run_id,
            selected_policy=recommendation.action_id,
            predicted_blocked_arrivals=metrics.total_blocked_arrivals,
            predicted_unsafe_excess=metrics.total_unsafe_excess,
            db_path=self.db_path,
        )
        log_alerts(run_id=run_id, alerts=alert_objects, db_path=self.db_path)

        manifest_path, manifest_hash = write_run_manifest(
            run_id=run_id,
            run_type="service_decision_recommendation",
            parameters={
                "request": request.model_dump(mode="json"),
                "resolved_scenario": scenario.model_dump(mode="json"),
            },
            input_paths=[
                self.scenario_service.data_dir / "base_instance",
                self.scenario_service.data_dir / "transitions",
            ],
            metrics={
                **metrics.model_dump(),
                "overall_alert_level": overall_level,
                "recommended_action": recommendation.action_id,
            },
            model_info={
                "simulation": "stochastic state-transition twin",
                "optimizer": "continuous network capacity LP",
                "decision_policy": "threshold-based human-review orchestration",
            },
            notes=[
                "Recommendation augments human operational judgment and is not autonomous.",
                "Synthetic Meridian reference case; not real hospital evidence.",
            ],
            manifest_dir=self.manifest_dir,
        )
        log_run_manifest(run_id, manifest_path, manifest_hash, db_path=self.db_path)

        return DecisionRecommendationResponse(
            run_id=run_id,
            overall_alert_level=overall_level,
            scenario=scenario,
            metrics=metrics,
            alerts=[
                AlertContract(
                    level=item.level,
                    trigger=item.trigger,
                    message=item.message,
                    timestamp=item.timestamp,
                )
                for item in alert_objects
            ],
            recommendation=recommendation,
            executive_brief=brief,
            metadata=ServiceMetadata(
                service=self.service_name,
                correlation_id=request.context.correlation_id,
                evidence=EvidenceReference(path=str(manifest_path), sha256=manifest_hash),
            ),
        )
