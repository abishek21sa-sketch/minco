from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config.paths import DATA_DIR
from src.services.container import build_service_container


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_release_evidence_is_hash_addressed_and_does_not_claim_production_approval(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MINCO_API_KEY", raising=False)
    correlation_id = "release-evidence-20260905"
    results_dir = tmp_path / "results"
    _write_json(results_dir / "contracts" / "service_contract_registry.json", {"schemas": {"test": "1"}})
    _write_json(
        results_dir / "validation" / "phase2_backend_acceptance_report.json",
        {"status": "passed"},
    )
    _write_json(
        results_dir.parent / "artifacts" / "product_runtime" / "latest_product_evidence.json",
        {"status": "passed"},
    )
    _write_json(results_dir.parent / "PRODUCT_V1_RELEASE_MANIFEST.json", {"release": "test"})
    status_path = results_dir.parent / "docs" / "execution" / "PROJECT_STATUS.md"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("test evidence", encoding="utf-8")
    services = build_service_container(
        data_dir=DATA_DIR,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        results_dir=results_dir,
    )

    response = TestClient(create_app(services)).get(
        "/v1/release-evidence",
        headers={"X-Correlation-ID": correlation_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert response.headers["X-Correlation-ID"] == correlation_id
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "VERIFIED"
    assert len(payload["build_fingerprint"]) == 64
    assert payload["gates"]["required_artifacts_hashed"] is True
    assert all(item["status"] == "VERIFIED" for item in payload["artifacts"])
    assert payload["human_review_required"] is True
    assert payload["autonomous_execution_permitted"] is False
    assert "does not establish" in payload["integrity_statement"]
