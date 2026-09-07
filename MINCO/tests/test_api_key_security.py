from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_configured_api_key_protects_operational_routes_but_not_liveness(monkeypatch) -> None:
    monkeypatch.setenv("MINCO_API_KEY", "enterprise-test-key-2026")
    client = TestClient(create_app())

    health = client.get("/health")
    unauthorized = client.get(
        "/v1/operational-state",
        headers={"X-Correlation-ID": "auth-failure-01"},
    )
    wrong_key = client.get("/v1/operational-state", headers={"X-API-Key": "wrong"})
    authorized = client.get(
        "/v1/operational-state",
        headers={"X-API-Key": "enterprise-test-key-2026"},
    )

    assert health.status_code == 200
    assert unauthorized.status_code == 401
    assert unauthorized.headers["X-Correlation-ID"] == "auth-failure-01"
    assert unauthorized.json()["detail"]["code"] == "authentication_required"
    assert wrong_key.status_code == 401
    assert authorized.status_code == 200
