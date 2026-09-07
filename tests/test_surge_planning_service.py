from __future__ import annotations

from pathlib import Path

from src.ai.demand_forecast_quantile import train_validate_demand_forecast
from src.config.loader import load_healthcare_instance
from src.contracts.common import ExecutionContext
from src.contracts.surge_plan import SurgePlanRequest
from src.data_engineering.synthetic_arrival_history import generate_synthetic_arrival_history
from src.services.scenario_service import ScenarioService
from src.services.surge_planning_service import SurgePlanningService


def test_surge_planning_service_couples_ai_ie_or_and_simulation(tmp_path: Path) -> None:
    instance = load_healthcare_instance("data")
    history = generate_synthetic_arrival_history(instance, n_days=220, seed=20260816)
    history_path = tmp_path / "history.csv"
    model_path = tmp_path / "model.joblib"
    history.to_csv(history_path, index=False)
    report = train_validate_demand_forecast(
        history,
        model_path=model_path,
        validation_path=tmp_path / "validation.json",
        prediction_path=tmp_path / "predictions.csv",
    )
    assert report["status"] == "passed"

    def runner(instance, parameters, n_replications):
        # Solver/simulation stub: the integration contract must preserve these
        # OR outputs and stochastic metrics without needing Gurobi in this unit test.
        return {
            "total_unsafe_excess": 0.0,
            "total_overflow_excess": 0.0,
            "max_utilization_ratio": 0.95,
            "num_unsafe_rows": 0.0,
            "total_blocked_arrivals": 1.0,
            "n_replications": n_replications,
            "model_status": 2,
            "solver_status_label": "OPTIMAL",
            "objective_value": 42.0,
            "total_surge_activated": 3.0,
            "total_elective_rejected": 1.0,
            "total_icu_transfer_load": 2.0,
            "solve_runtime_seconds": 0.01,
        }

    db_path = tmp_path / "audit.db"
    manifest_dir = tmp_path / "manifests"
    scenario_service = ScenarioService(
        data_dir=Path("data"),
        runner=runner,
        db_path=db_path,
        manifest_dir=manifest_dir,
    )
    service = SurgePlanningService(
        scenario_service,
        data_dir=Path("data"),
        history_path=history_path,
        model_path=model_path,
        db_path=db_path,
        manifest_dir=manifest_dir,
    )
    response = service.plan(
        SurgePlanRequest(
            scenario_id="baseline",
            n_replications=3,
            context=ExecutionContext(source="test_surge_planning"),
        )
    )

    assert len(response.forecasts) == 12
    assert len(response.resource_pressure_p50) == 6
    assert len(response.resource_pressure_p95) == 6
    assert response.forecast_summary.network_p95 >= response.forecast_summary.network_p50
    assert response.forecast_summary.effective_optimizer_demand_multiplier > 0
    assert response.forecast_summary.planning_quantile == "p95"
    assert (
        response.forecast_summary.planning_to_seasonal_naive_ratio
        == response.forecast_summary.p95_to_seasonal_naive_ratio
    )
    assert response.scenario_evaluation.metrics.solver_status_label == "OPTIMAL"
    assert response.optimized_actions.total_surge_activated == 3.0
    assert response.optimized_actions.evidence_label == "OPTIMIZED"
    assert response.decision.human_review_required is True
    assert response.metadata.evidence is not None
    assert Path(response.metadata.evidence.path).exists()


def test_surge_planning_quantile_changes_optimizer_guardrail(tmp_path: Path) -> None:
    instance = load_healthcare_instance("data")
    history = generate_synthetic_arrival_history(instance, n_days=220, seed=20260816)
    history_path = tmp_path / "history.csv"
    model_path = tmp_path / "model.joblib"
    history.to_csv(history_path, index=False)
    train_validate_demand_forecast(
        history,
        model_path=model_path,
        validation_path=tmp_path / "validation.json",
        prediction_path=tmp_path / "predictions.csv",
    )

    seen = []

    def runner(instance, parameters, n_replications):
        seen.append(parameters.demand_surge_multiplier)
        return {
            "total_unsafe_excess": 0.0,
            "total_overflow_excess": 0.0,
            "max_utilization_ratio": 0.9,
            "num_unsafe_rows": 0.0,
            "total_blocked_arrivals": 0.0,
            "n_replications": n_replications,
            "model_status": 2,
            "solver_status_label": "OPTIMAL",
            "objective_value": 1.0,
            "total_surge_activated": 0.0,
            "total_elective_rejected": 0.0,
            "total_icu_transfer_load": 0.0,
            "solve_runtime_seconds": 0.01,
        }

    scenario_service = ScenarioService(
        data_dir=Path("data"),
        runner=runner,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
    )
    service = SurgePlanningService(
        scenario_service,
        data_dir=Path("data"),
        history_path=history_path,
        model_path=model_path,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
    )
    p50 = service.plan(
        SurgePlanRequest(scenario_id="baseline", planning_quantile="p50", n_replications=2)
    )
    p95 = service.plan(
        SurgePlanRequest(scenario_id="baseline", planning_quantile="p95", n_replications=2)
    )
    assert p50.forecast_summary.planning_quantile == "p50"
    assert p95.forecast_summary.planning_quantile == "p95"
    assert (
        p95.forecast_summary.effective_optimizer_demand_multiplier
        >= p50.forecast_summary.effective_optimizer_demand_multiplier
    )
    assert seen[1] >= seen[0]
