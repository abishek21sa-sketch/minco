from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_health_endpoint_starts_without_solver() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "minco-api"


def test_readiness_reports_data_state() -> None:
    client = TestClient(create_app())
    response = client.get("/readiness")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data_ready"] is True


def test_solver_endpoint_fails_transparently_without_gurobi() -> None:
    client = TestClient(create_app())
    response = client.post("/what-if", json={})
    assert response.status_code in {200, 503}
    if response.status_code == 503:
        payload = response.json()["detail"]
        assert payload["code"] == "solver_runtime_unavailable"


def test_empty_audit_log_is_safe() -> None:
    client = TestClient(create_app())
    response = client.get("/audit-log?n=1")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_model_governance_endpoint_returns_saved_report(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "predictive_model_governance_summary.json"
    report_path.write_text(
        '{"status":"passed","status_counts":{"approved_for_reference_case":1}}',
        encoding="utf-8",
    )
    monkeypatch.setattr("src.api.main.PREDICTIVE_GOVERNANCE_PATH", report_path)
    client = TestClient(create_app())
    response = client.get("/model-governance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "passed"
    assert "governance_note" in payload


def test_validation_status_marks_level2_as_historical_when_report_exists(
    tmp_path, monkeypatch
) -> None:
    report_path = tmp_path / "level2_completion_report.json"
    report_path.write_text('{"status":"passed"}', encoding="utf-8")
    monkeypatch.setattr("src.api.main.LEVEL2_COMPLETION_PATH", report_path)
    client = TestClient(create_app())
    response = client.get("/validation-status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "passed"
    assert "historical engineering evidence" in payload["finalization_note"]


def test_finalization_status_endpoint_returns_saved_report(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "finalization_phase1_validation_report.json"
    report_path.write_text('{"status":"passed","work_package":"phase1"}', encoding="utf-8")
    monkeypatch.setattr("src.api.main.FINALIZATION_PHASE1_REPORT_PATH", report_path)
    client = TestClient(create_app())
    response = client.get("/finalization-status")
    assert response.status_code == 200
    assert response.json()["work_package"] == "phase1"
