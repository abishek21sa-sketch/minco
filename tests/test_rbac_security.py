from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config.paths import DATA_DIR
from src.contracts.scenario import ScenarioParameters
from src.services.container import build_service_container


def fake_runner(instance, parameters: ScenarioParameters, n_replications: int):
    surge = parameters.demand_surge_multiplier
    return {
        "total_unsafe_excess": 2.0 * surge,
        "total_overflow_excess": 0.5 * surge,
        "max_utilization_ratio": 0.9 * surge,
        "num_unsafe_rows": 1.0,
        "total_blocked_arrivals": 2.0 * surge,
        "n_replications": n_replications,
        "model_status": 2,
        "solve_runtime_seconds": 0.01,
    }


def test_identity_and_role_based_review_controls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MINCO_API_KEY", raising=False)
    monkeypatch.setenv(
        "MINCO_API_KEYS",
        json.dumps(
            {
                "viewer-secret": {"subject": "command-center-viewer", "role": "viewer"},
                "analyst-secret": {"subject": "capacity-analyst", "role": "analyst"},
                "reviewer-secret": {"subject": "operations-reviewer", "role": "reviewer"},
            }
        ),
    )
    services = build_service_container(
        data_dir=DATA_DIR,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        scenario_runner=fake_runner,
    )
    client = TestClient(create_app(services))

    assert client.get("/health").status_code == 200
    assert client.get("/v1/scenarios").status_code == 401

    viewer_headers = {"X-API-Key": "viewer-secret"}
    viewer_identity = client.get("/v1/identity", headers=viewer_headers)
    assert viewer_identity.status_code == 200
    assert viewer_identity.json()["subject"] == "command-center-viewer"
    assert viewer_identity.json()["role"] == "viewer"
    assert viewer_identity.json()["authorization_enforced"] is True
    assert viewer_identity.json()["autonomous_execution_permitted"] is False

    denied_evaluation = client.post(
        "/v1/scenarios/evaluate",
        headers=viewer_headers,
        json={"scenario_id": "flu_surge", "n_replications": 2},
    )
    assert denied_evaluation.status_code == 403
    assert denied_evaluation.json()["detail"]["code"] == "insufficient_role"
    assert denied_evaluation.json()["detail"]["required_role"] == "analyst"

    analyst_headers = {"Authorization": "Bearer analyst-secret"}
    evaluation = client.post(
        "/v1/scenarios/evaluate",
        headers=analyst_headers,
        json={"scenario_id": "flu_surge", "n_replications": 2},
    )
    assert evaluation.status_code == 200
    run_id = evaluation.json()["run_id"]

    denied_review = client.post(
        f"/v1/audit/{run_id}/review",
        headers=analyst_headers,
        json={
            "reviewed_by": "capacity-analyst",
            "decision": "DEFERRED",
            "comment": "Needs review.",
        },
    )
    assert denied_review.status_code == 403
    assert denied_review.json()["detail"]["required_role"] == "reviewer"

    reviewer_headers = {"X-API-Key": "reviewer-secret"}
    reviewer_identity = client.get("/v1/identity", headers=reviewer_headers)
    assert reviewer_identity.json()["role"] == "reviewer"
    review = client.post(
        f"/v1/audit/{run_id}/review",
        headers=reviewer_headers,
        json={
            "reviewed_by": "operations-reviewer",
            "decision": "DEFERRED",
            "comment": "Review role verified before local disposition.",
        },
    )
    assert review.status_code == 200
    assert review.json()["autonomous_execution_permitted"] is False
