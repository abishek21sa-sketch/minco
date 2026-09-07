from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_validation_status_endpoint_returns_level2_report(tmp_path, monkeypatch) -> None:
    report = tmp_path / "level2_completion_report.json"
    report.write_text(
        '{"status":"passed","level_2_disposition":"complete_and_level_3_authorized"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("src.api.main.LEVEL2_COMPLETION_PATH", report)
    catalog = tmp_path / "data_asset_registry_summary.json"
    catalog.write_text('{"status":"passed","asset_count":17}', encoding="utf-8")
    monkeypatch.setattr("src.api.main.DATA_ASSET_REGISTRY_SUMMARY_PATH", catalog)
    client = TestClient(create_app())
    response = client.get("/validation-status")
    assert response.status_code == 200
    assert response.json()["status"] == "passed"
    catalog_response = client.get("/evidence-catalog")
    assert catalog_response.status_code == 200
    assert catalog_response.json()["asset_count"] == 17
