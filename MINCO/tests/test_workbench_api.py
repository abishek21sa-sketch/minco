from __future__ import annotations

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


def _write_release_evidence(results_dir: Path) -> None:
    files = {
        results_dir / "contracts" / "service_contract_registry.json": '{"schemas":{"test":"1"}}',
        results_dir / "validation" / "phase2_backend_acceptance_report.json": '{"status":"passed"}',
        results_dir.parent
        / "artifacts"
        / "product_runtime"
        / "latest_product_evidence.json": '{"status":"passed"}',
        results_dir.parent / "PRODUCT_V1_RELEASE_MANIFEST.json": '{"release":"test"}',
        results_dir.parent / "docs" / "execution" / "PROJECT_STATUS.md": "test evidence",
    }
    for path, contents in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def test_workbench_evaluates_counterfactual_and_records_review(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_release_evidence(results_dir)
    services = build_service_container(
        data_dir=DATA_DIR,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        results_dir=results_dir,
        scenario_runner=fake_runner,
    )
    client = TestClient(create_app(services))
    context = {"correlation_id": "workbench-test-01", "source": "command_center_workbench"}

    baseline = client.post(
        "/v1/scenarios/evaluate",
        json={"scenario_id": "baseline", "n_replications": 2, "context": context},
    )
    counterfactual = client.post(
        "/v1/scenarios/evaluate",
        json={"scenario_id": "flu_surge", "n_replications": 2, "context": context},
    )

    assert baseline.status_code == 200
    assert counterfactual.status_code == 200
    assert (
        counterfactual.json()["metrics"]["total_unsafe_excess"]
        > baseline.json()["metrics"]["total_unsafe_excess"]
    )

    run_id = counterfactual.json()["run_id"]
    review = client.post(
        f"/v1/audit/{run_id}/review",
        json={
            "reviewed_by": "command-center-lead",
            "decision": "DEFERRED",
            "comment": "Compare against current operating facts before routing for local review.",
        },
    )

    assert review.status_code == 200
    assert review.json()["run_id"] == run_id
    assert review.json()["autonomous_execution_permitted"] is False
    assert client.get("/v1/audit/integrity").json()["status"] == "VALID"

    packet = client.get(
        f"/v1/decision-packets/{run_id}",
        headers={"X-Correlation-ID": "packet-test-01"},
    )

    assert packet.status_code == 200
    payload = packet.json()
    assert payload["packet_id"] == f"packet_{run_id}"
    assert payload["what_if_run"]["run_id"] == run_id
    assert payload["manifest"]["run_id"] == run_id
    assert payload["governance"]["packet_ready"] is True
    assert payload["governance"]["review_status"] == "DEFERRED"
    assert payload["governance"]["autonomous_execution_permitted"] is False
    assert payload["audit_integrity"]["status"] == "VALID"
    assert payload["release_evidence"]["status"] == "VERIFIED"

    assert client.get("/v1/decision-packets/unknown-run").status_code == 404
