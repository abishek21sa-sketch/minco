from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_release_readiness_is_explicitly_blocked_for_local_reference_package(monkeypatch) -> None:
    monkeypatch.delenv("MINCO_API_KEY", raising=False)
    correlation_id = "release-ready-20260904"

    response = TestClient(create_app()).get(
        "/v1/release-readiness",
        headers={"X-Correlation-ID": correlation_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert response.headers["X-Correlation-ID"] == correlation_id
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "BLOCKED"
    assert payload["release_readiness"] == "NOT_FOR_PRODUCTION"
    assert payload["checks"]["synthetic_case_explicitly_labeled"] is True
    assert payload["checks"]["autonomous_execution_disabled"] is True
    assert payload["security_controls"]["security_mode"] == "LOCAL_REFERENCE_CASE_NO_AUTH"
    assert any("live ADT/EHR" in reason for reason in payload["blocking_reasons"])
