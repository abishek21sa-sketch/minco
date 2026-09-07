"""Integrated AI -> IE -> OR -> simulation surge-planning service."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ai.demand_forecast_quantile import (
    DEFAULT_MODEL_PATH,
    forecast_next_day,
    load_demand_forecast_bundle,
)
from src.config.loader import load_healthcare_instance
from src.config.paths import AUDIT_DB_PATH, DATA_DIR, RESULTS_DIR
from src.contracts.common import EvidenceReference, ServiceMetadata
from src.contracts.surge_plan import (
    DemandForecastItem,
    DemandForecastSummary,
    OptimizedActionSummary,
    ResourcePressureItem,
    SurgeDecision,
    SurgePlanRequest,
    SurgePlanResponse,
)
from src.contracts.scenario import ScenarioEvaluationRequest, ScenarioParameters
from src.ie.healthcare_flow import build_resource_pressure_projection
from src.services.scenario_service import ScenarioService
from src.storage.audit_repository import (
    log_decision_run,
    log_recommendation,
    log_run_manifest,
)
from src.validation.run_manifest import RUN_MANIFEST_DIR, write_run_manifest

DEFAULT_HISTORY_PATH = RESULTS_DIR / "reference_history" / "synthetic_arrival_history.csv"


class SurgePlanningNotReadyError(RuntimeError):
    pass


def _resource_items(frame: pd.DataFrame) -> list[ResourcePressureItem]:
    return [
        ResourcePressureItem(
            hospital_id=str(row.hospital_id),
            resource=str(row.resource),
            source_quantile=str(row.source_quantile),
            expected_census=float(row.expected_census),
            base_capacity=float(row.base_capacity),
            safe_utilization=float(row.safe_utilization),
            safe_capacity=float(row.safe_capacity),
            projected_utilization=float(row.projected_utilization),
            safe_capacity_gap=float(row.safe_capacity_gap),
            above_safe_capacity=bool(row.above_safe_capacity),
        )
        for row in frame.itertuples(index=False)
    ]


def _decision_from_plan(
    *,
    p95_pressure: pd.DataFrame,
    scenario_metrics: Any,
) -> SurgeDecision:
    above_safe = int(p95_pressure["above_safe_capacity"].sum())
    max_p95_util = float(p95_pressure["projected_utilization"].max())
    solver_status = scenario_metrics.solver_status_label

    if solver_status not in {None, "OPTIMAL"}:
        return SurgeDecision(
            action_id="solver_result_requires_review",
            title="Do not operationalize the plan until solver status is reviewed",
            priority="immediate",
            rationale=[f"Solver status is {solver_status or 'unknown'}, not a verified optimal result."],
        )

    if scenario_metrics.total_unsafe_excess > 0 or scenario_metrics.total_overflow_excess > 0:
        return SurgeDecision(
            action_id="execute_optimized_surge_response_after_human_review",
            title="Review and execute the optimized surge-response plan",
            priority="immediate",
            rationale=[
                "The optimized stochastic evaluation still projects unsafe or overflow pressure.",
                f"P95 IE capacity projection has {above_safe} resource rows above safe capacity.",
                f"Maximum P95 Little's-Law utilization is {max_p95_util:.2f}x base capacity.",
            ],
        )

    if above_safe > 0:
        return SurgeDecision(
            action_id="preposition_capacity_and_monitor",
            title="Pre-position capacity and monitor the forecast guardrail",
            priority="monitor",
            rationale=[
                "The optimized stochastic evaluation is controlled, but the P95 demand translation exceeds one or more safe-capacity thresholds.",
                f"{above_safe} hospital-resource rows exceed safe capacity in the IE projection.",
            ],
        )

    return SurgeDecision(
        action_id="continue_coordinated_operation",
        title="Continue coordinated operation under the optimized policy",
        priority="routine",
        rationale=[
            "The forecast-guarded optimization remains within modeled safe-capacity conditions.",
            "No P95 resource-pressure row exceeds its configured safe-capacity threshold.",
        ],
    )


class SurgePlanningService:
    """Couple probabilistic arrivals, IE capacity math, OR, and simulation."""

    service_name = "surge_planning_service"

    def __init__(
        self,
        scenario_service: ScenarioService,
        *,
        data_dir: Path = DATA_DIR,
        history_path: Path = DEFAULT_HISTORY_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        db_path: Path = AUDIT_DB_PATH,
        manifest_dir: Path = RUN_MANIFEST_DIR,
    ) -> None:
        self.scenario_service = scenario_service
        self.data_dir = Path(data_dir)
        self.history_path = Path(history_path)
        self.model_path = Path(model_path)
        self.db_path = Path(db_path)
        self.manifest_dir = Path(manifest_dir)

    def _load_forecast_inputs(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not self.history_path.exists():
            raise SurgePlanningNotReadyError(
                f"Synthetic reference history not found at {self.history_path}. "
                "Run python -m src.system.run_finalization_phase1_validation."
            )
        if not self.model_path.exists():
            raise SurgePlanningNotReadyError(
                f"Governed arrival model not found at {self.model_path}. "
                "Run python -m src.system.run_finalization_phase1_validation."
            )
        history = pd.read_csv(self.history_path)
        bundle = load_demand_forecast_bundle(self.model_path)
        return history, bundle

    def plan(self, request: SurgePlanRequest) -> SurgePlanResponse:
        history, bundle = self._load_forecast_inputs()
        forecast = forecast_next_day(history, bundle=bundle)
        instance = load_healthcare_instance(self.data_dir)

        base_scenario = self.scenario_service.resolve_scenario(
            ScenarioEvaluationRequest(
                scenario_id=request.scenario_id,
                overrides=request.overrides,
                n_replications=request.n_replications,
                context=request.context,
            )
        )
        scenario_multiplier = float(base_scenario.parameters.demand_surge_multiplier)

        forecast_p50 = forecast.copy()
        forecast_p95 = forecast.copy()
        forecast_p50["scenario_adjusted_p50"] = forecast_p50["predicted_p50"] * scenario_multiplier
        forecast_p95["scenario_adjusted_p95"] = forecast_p95["predicted_p95"] * scenario_multiplier

        pressure_p50 = build_resource_pressure_projection(
            instance,
            forecast_p50.rename(columns={"scenario_adjusted_p50": "planning_arrivals"}),
            quantile_column="planning_arrivals",
        )
        pressure_p50["source_quantile"] = "scenario_adjusted_p50"
        pressure_p95 = build_resource_pressure_projection(
            instance,
            forecast_p95.rename(columns={"scenario_adjusted_p95": "planning_arrivals"}),
            quantile_column="planning_arrivals",
        )
        pressure_p95["source_quantile"] = "scenario_adjusted_p95"

        network_p05 = float(forecast["predicted_p05"].sum())
        network_p50 = float(forecast["predicted_p50"].sum())
        network_p95 = float(forecast["predicted_p95"].sum())
        seasonal_naive_total = float(forecast["seasonal_naive"].sum())
        if seasonal_naive_total <= 0:
            raise SurgePlanningNotReadyError("Seasonal-naive reference demand is zero")
        p95_ratio = network_p95 / seasonal_naive_total
        planning_total = network_p95 if request.planning_quantile == "p95" else network_p50
        planning_ratio = planning_total / seasonal_naive_total
        effective_multiplier = min(5.0, max(0.0, planning_ratio * scenario_multiplier))

        effective_parameters = ScenarioParameters(
            icu_bed_delta=base_scenario.parameters.icu_bed_delta,
            transfer_capacity_multiplier=base_scenario.parameters.transfer_capacity_multiplier,
            demand_surge_multiplier=effective_multiplier,
        )
        scenario_response = self.scenario_service.evaluate(
            ScenarioEvaluationRequest(
                scenario_id="custom",
                overrides=effective_parameters,
                n_replications=request.n_replications,
                context=request.context.model_copy(
                    update={"source": f"{request.context.source}:forecast_guarded_surge_plan"}
                ),
            )
        )

        metrics = scenario_response.metrics
        optimized_actions = OptimizedActionSummary(
            solver_status=metrics.solver_status_label,
            objective_value=metrics.objective_value,
            total_surge_activated=metrics.total_surge_activated,
            total_elective_rejected=metrics.total_elective_rejected,
            total_icu_transfer_load=metrics.total_icu_transfer_load,
        )
        decision = _decision_from_plan(p95_pressure=pressure_p95, scenario_metrics=metrics)

        forecast_date = str(forecast["forecast_date"].iloc[0])
        summary = DemandForecastSummary(
            forecast_date=forecast_date,
            network_p05=network_p05,
            network_p50=network_p50,
            network_p95=network_p95,
            seasonal_naive_total=seasonal_naive_total,
            p95_to_seasonal_naive_ratio=p95_ratio,
            planning_quantile=request.planning_quantile,
            planning_to_seasonal_naive_ratio=planning_ratio,
            scenario_demand_multiplier=scenario_multiplier,
            effective_optimizer_demand_multiplier=effective_multiplier,
        )

        alert_level = {"immediate": "RED", "monitor": "YELLOW", "routine": "GREEN"}[decision.priority]
        run_id = log_decision_run(
            scenario=f"surge_plan:{base_scenario.scenario_id}",
            overall_alert=alert_level,
            pipeline_mode=request.context.source,
            runtime_seconds=metrics.solve_runtime_seconds,
            db_path=self.db_path,
        )
        log_recommendation(
            run_id=run_id,
            selected_policy=decision.action_id,
            predicted_blocked_arrivals=metrics.total_blocked_arrivals,
            predicted_unsafe_excess=metrics.total_unsafe_excess,
            db_path=self.db_path,
        )

        manifest_path, manifest_hash = write_run_manifest(
            run_id=run_id,
            run_type="integrated_surge_plan",
            parameters={
                "request": request.model_dump(mode="json"),
                "forecast_summary": summary.model_dump(mode="json"),
                "effective_scenario_parameters": effective_parameters.model_dump(mode="json"),
            },
            input_paths=[
                self.history_path,
                self.model_path,
                self.data_dir / "base_instance",
                self.data_dir / "transitions",
            ],
            metrics={
                **metrics.model_dump(),
                "p95_resource_rows_above_safe_capacity": int(
                    pressure_p95["above_safe_capacity"].sum()
                ),
                "max_p95_projected_utilization": float(
                    pressure_p95["projected_utilization"].max()
                ),
                "recommended_action": decision.action_id,
                "scenario_evaluation_run_id": scenario_response.run_id,
            },
            model_info={
                "ai": "quantile_gradient_boosting_next_day_arrivals",
                "ai_validation": "chronological_synthetic_replay_only",
                "industrial_engineering": "absorbing_markov_expected_resource_days_plus_littles_law",
                "optimization": "continuous_network_capacity_lp_gurobi",
                "simulation": "seeded_stochastic_state_transition_twin",
            },
            notes=[
                "AI forecast is validated only on generated synthetic historical replay.",
                "Little's-Law capacity translation is a steady-state engineering approximation.",
                "Optimization and simulation results are modeled decision support, not realized outcomes.",
                "Human review is mandatory before operational action.",
            ],
            manifest_dir=self.manifest_dir,
        )
        log_run_manifest(run_id, manifest_path, manifest_hash, db_path=self.db_path)

        return SurgePlanResponse(
            run_id=run_id,
            forecast_summary=summary,
            forecasts=[DemandForecastItem(**row) for row in forecast.to_dict("records")],
            resource_pressure_p50=_resource_items(pressure_p50),
            resource_pressure_p95=_resource_items(pressure_p95),
            scenario_evaluation=scenario_response,
            optimized_actions=optimized_actions,
            decision=decision,
            metadata=ServiceMetadata(
                service=self.service_name,
                correlation_id=request.context.correlation_id,
                evidence=EvidenceReference(path=str(manifest_path), sha256=manifest_hash),
                warnings=[
                    "Synthetic reference-case planning only; no real-hospital deployment approval.",
                ],
            ),
        )
