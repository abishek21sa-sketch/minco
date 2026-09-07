from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_operational_history_endpoint_returns_replay_provenance_and_freshness() -> None:
    correlation_id = "history-smoke-20260904"
    response = TestClient(create_app()).get(
        "/v1/operational-history",
        headers={"X-Correlation-ID": correlation_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert response.headers["X-Correlation-ID"] == correlation_id
    assert payload["schema_version"] == "1.0"
    assert payload["event_count"] >= 459
    assert payload["freshness_status"] == "REPLAY_ONLY"
    assert payload["live_feed_connected"] is False
    assert payload["evidence"]["evidence_label"] == "REPLAYED"
    assert payload["event_type_counts"]["arrival"] > 0
