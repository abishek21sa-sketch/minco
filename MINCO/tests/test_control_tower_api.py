from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_control_tower_endpoint_returns_enterprise_snapshot() -> None:
    response = TestClient(create_app()).get("/v1/control-tower")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["decision"]["algorithm"] == "FLOW-CVaR"
    assert payload["governance"]["autonomous_execution_permitted"] is False
    assert "patient_flow" in payload
    assert payload["operational_history"]["event_count"] >= 459
    assert payload["operational_history"]["live_feed_connected"] is False
    assert "diagnostics" in payload
    assert "evidence" in payload
