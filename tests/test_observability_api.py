from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.observability.metrics import API_METRICS


def test_system_contracts_return_and_propagate_correlation_id() -> None:
    client = TestClient(create_app())
    correlation_id = "ops-cycle-20260904"

    health = client.get("/health", headers={"X-Correlation-ID": correlation_id})
    readiness = client.get("/readiness", headers={"X-Correlation-ID": correlation_id})
    state = client.get("/v1/operational-state", headers={"X-Correlation-ID": correlation_id})

    assert health.status_code == 200
    assert health.headers["X-Correlation-ID"] == correlation_id
    assert health.json()["correlation_id"] == correlation_id
    assert readiness.status_code == 200
    assert readiness.json()["schema_version"] == "1.0"
    assert readiness.json()["autonomous_execution_permitted"] is False
    assert state.status_code == 200
    assert state.json()["metadata"]["correlation_id"] == correlation_id


def test_request_telemetry_is_structured_and_excludes_query_payload(caplog) -> None:
    caplog.set_level(logging.INFO, logger="minco.api")
    correlation_id = "telemetry-check-01"

    response = TestClient(create_app()).get(
        "/health?secret=should-not-be-logged",
        headers={"X-Correlation-ID": correlation_id},
    )

    assert response.status_code == 200
    events = [
        record.minco_event
        for record in caplog.records
        if getattr(record, "name", "") == "minco.api" and hasattr(record, "minco_event")
    ]
    assert events
    event = events[-1]
    assert event["event"] == "api_request_completed"
    assert event["correlation_id"] == correlation_id
    assert event["path"] == "/health"
    assert "secret" not in str(event)


def test_metrics_are_typed_low_cardinality_and_prometheus_compatible() -> None:
    API_METRICS.reset()
    client = TestClient(create_app())

    client.get("/health")
    client.get("/v1/scenarios/baseline")
    typed = client.get(
        "/v1/observability/metrics",
        headers={"X-Correlation-ID": "metrics-contract-01"},
    )
    prometheus = client.get("/metrics")

    assert typed.status_code == 200
    payload = typed.json()
    assert payload["schema_version"] == "1.0"
    assert payload["request_count"] >= 2
    assert payload["route_counts"]["GET /health"] == 1
    assert payload["data_logging_policy"]["request_payloads_logged"] is False
    assert prometheus.status_code == 200
    assert "minco_api_requests_total" in prometheus.text
    assert "secret" not in prometheus.text
