from __future__ import annotations

from pathlib import Path

import pytest

from src.config.paths import DATA_DIR
from src.contracts.common import ExecutionContext
from src.contracts.decision import DecisionRecommendationRequest
from src.contracts.scenario import ScenarioEvaluationRequest, ScenarioParameters
from src.services.container import build_service_container
from src.services.operational_state_service import OperationalStateService
from src.services.scenario_service import ScenarioNotFoundError, ScenarioService
from src.storage.audit_repository import get_recent_audit_runs


def fake_runner(instance, parameters: ScenarioParameters, n_replications: int):
    assert instance.hospitals.df.shape[0] == 3
    return {
        "total_unsafe_excess": max(0.0, 4.0 * parameters.demand_surge_multiplier),
        "total_overflow_excess": 1.0,
        "max_utilization_ratio": 1.05,
        "num_unsafe_rows": 2.0,
        "total_blocked_arrivals": 3.0,
        "n_replications": n_replications,
        "model_status": 2,
        "solve_runtime_seconds": 0.01,
    }


def test_operational_state_service_reconstructs_reference_snapshot() -> None:
    response = OperationalStateService(DATA_DIR).get_reference_snapshot(
        ExecutionContext(source="test")
    )
    assert response.state.hospital_count == 3
    assert response.state.cohort_count == 4
    assert response.state.horizon_days == 7
    assert response.state.allowed_transfer_lanes == 6
    assert response.state.total_transfer_capacity_per_day == 28.0
    assert response.state.input_fingerprints["base_instance"]


def test_scenario_catalog_and_unknown_scenario() -> None:
    service = ScenarioService(DATA_DIR, runner=fake_runner)
    assert {item.scenario_id for item in service.list_scenarios()} >= {
        "baseline",
        "flu_surge",
        "transfer_failure",
    }
    with pytest.raises(ScenarioNotFoundError):
        service.get_scenario("not-a-scenario")


def test_scenario_service_persists_audited_contract_response(tmp_path: Path) -> None:
    service = ScenarioService(
        DATA_DIR,
        runner=fake_runner,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
    )
    response = service.evaluate(
        ScenarioEvaluationRequest(
            scenario_id="custom",
            overrides=ScenarioParameters(demand_surge_multiplier=1.2),
            n_replications=3,
            context=ExecutionContext(source="test_scenario_service"),
        )
    )
    assert response.metrics.model_status == 2
    assert response.metadata.evidence is not None
    assert Path(response.metadata.evidence.path).exists()
    rows = get_recent_audit_runs(n=10, db_path=tmp_path / "audit.db")
    assert rows[0]["run_id"] == response.run_id
    assert rows[0]["manifest_sha256"] == response.metadata.evidence.sha256


def test_decision_orchestration_returns_human_review_action(tmp_path: Path) -> None:
    container = build_service_container(
        data_dir=DATA_DIR,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        scenario_runner=fake_runner,
    )
    response = container.decisions.recommend(
        DecisionRecommendationRequest(
            scenario_id="baseline",
            n_replications=3,
            context=ExecutionContext(source="test_decision_service"),
        )
    )
    assert response.metrics.model_status == 2
    assert response.recommendation.human_review_required is True
    assert response.overall_alert_level in {"GREEN", "YELLOW", "RED"}
    assert response.metadata.evidence is not None
