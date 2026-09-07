from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_command_center_root_is_served_by_governed_api(monkeypatch) -> None:
    monkeypatch.delenv("MINCO_API_KEY", raising=False)

    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "MINCO Command Center" in response.text
    assert "/v1/control-tower" in response.text
    assert "Operations Workbench" in response.text
    assert "/v1/scenarios/evaluate" in response.text
    assert "/v1/decision-packets/" in response.text
    assert "/v1/identity" in response.text
    assert "Build evidence packet" in response.text
    assert "Record review" in response.text
    assert "Human-gated by design" in response.text


def test_command_center_shell_remains_public_when_api_key_is_enabled(monkeypatch) -> None:
    monkeypatch.setenv("MINCO_API_KEY", "ui-test-key")

    client = TestClient(create_app())

    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/v1/release-evidence").status_code == 401
