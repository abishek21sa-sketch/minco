from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config.paths import DATA_DIR
from src.contracts.scenario import ScenarioParameters
from src.services.container import build_service_container


def fake_runner(instance, parameters: ScenarioParameters, n_replications: int):
    return {
        "total_unsafe_excess": 4.0,
        "total_overflow_excess": 1.0,
        "max_utilization_ratio": 1.05,
        "num_unsafe_rows": 2.0,
        "total_blocked_arrivals": 3.0,
        "n_replications": n_replications,
        "model_status": 2,
        "solve_runtime_seconds": 0.01,
    }


def client_for(tmp_path: Path) -> TestClient:
    services = build_service_container(
        data_dir=DATA_DIR,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        scenario_runner=fake_runner,
    )
    return TestClient(create_app(services))


def test_v1_operational_state_and_scenario_catalog(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    state = client.get("/v1/operational-state")
    scenarios = client.get("/v1/scenarios")
    assert state.status_code == 200
    assert state.json()["metadata"]["contract_version"] == "1.0"
    assert state.json()["state"]["hospital_count"] == 3
    assert scenarios.status_code == 200
    assert any(item["scenario_id"] == "baseline" for item in scenarios.json())


def test_v1_scenario_evaluation_returns_typed_evidence(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    response = client.post(
        "/v1/scenarios/evaluate",
        json={
            "scenario_id": "baseline",
            "n_replications": 3,
            "context": {"source": "api_contract_test"},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["model_status"] == 2
    assert len(payload["metadata"]["evidence"]["sha256"]) == 64


def test_v1_decision_and_legacy_alias_share_services(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    decision = client.post(
        "/v1/decisions/recommend",
        json={"scenario_id": "baseline", "n_replications": 3},
    )
    legacy = client.post("/what-if", json={"n_replications": 3})
    assert decision.status_code == 200
    assert decision.json()["recommendation"]["human_review_required"] is True
    assert legacy.status_code == 200
    assert legacy.json()["model_status"] == 2
    assert legacy.json()["manifest"]["sha256"]
