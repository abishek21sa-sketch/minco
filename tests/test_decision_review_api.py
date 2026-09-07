from __future__ import annotations

import sqlite3
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


def test_human_review_is_persisted_and_never_authorizes_execution(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    recommendation = client.post(
        "/v1/decisions/recommend",
        json={"scenario_id": "baseline", "n_replications": 3},
    )
    assert recommendation.status_code == 200
    run_id = recommendation.json()["run_id"]

    review = client.post(
        f"/v1/audit/{run_id}/review",
        json={
            "reviewed_by": "operations.lead",
            "decision": "ACCEPTED_FOR_OPERATIONS_REVIEW",
            "comment": "Reconciled against the current command-center facts; route for local approval.",
        },
    )
    assert review.status_code == 200
    payload = review.json()
    assert payload["run_id"] == run_id
    assert payload["decision"] == "ACCEPTED_FOR_OPERATIONS_REVIEW"
    assert payload["autonomous_execution_permitted"] is False
    assert payload["review_id"].startswith("review_")
    assert payload["hash_algorithm"] == "SHA-256"
    assert payload["previous_hash"] == "GENESIS"
    assert len(payload["event_hash"]) == 64

    history = client.get(f"/v1/audit/{run_id}/reviews")
    recent = client.get("/v1/audit/reviews?n=5")
    assert history.status_code == 200
    assert recent.status_code == 200
    assert history.json()[0]["review_id"] == payload["review_id"]
    assert recent.json()[0]["run_id"] == run_id

    integrity = client.get("/v1/audit/integrity")
    assert integrity.status_code == 200
    assert integrity.json()["status"] == "VALID"
    assert integrity.json()["review_count"] == 1
    assert integrity.json()["head_hash"] == payload["event_hash"]

    tower = client.get("/v1/control-tower")
    assert tower.status_code == 200
    workflow = tower.json()["governance"]["review_workflow"]
    assert workflow["recent_review_count"] == 1
    assert workflow["latest_review"]["review_id"] == payload["review_id"]


def test_review_requires_a_known_decision_run_and_governed_disposition(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    missing = client.post(
        "/v1/audit/run_missing/review",
        json={
            "reviewed_by": "operations.lead",
            "decision": "DEFERRED",
            "comment": "Need current operating facts.",
        },
    )
    invalid = client.post(
        "/v1/audit/run_missing/review",
        json={
            "reviewed_by": "operations.lead",
            "decision": "APPROVED_FOR_EXECUTION",
            "comment": "This disposition is outside the governed workflow.",
        },
    )
    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_audit_integrity_detects_tampering(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    recommendation = client.post(
        "/v1/decisions/recommend",
        json={"scenario_id": "baseline", "n_replications": 3},
    )
    run_id = recommendation.json()["run_id"]
    review = client.post(
        f"/v1/audit/{run_id}/review",
        json={
            "reviewed_by": "operations.lead",
            "decision": "DEFERRED",
            "comment": "Awaiting current operating facts.",
        },
    )
    assert review.status_code == 200

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.execute(
            "UPDATE decision_reviews SET comment = ? WHERE review_id = ?",
            ("tampered", review.json()["review_id"]),
        )

    integrity = client.get("/v1/audit/integrity")
    assert integrity.status_code == 200
    assert integrity.json()["status"] == "INVALID"
    assert any(item["code"] == "event_hash_mismatch" for item in integrity.json()["violations"])
