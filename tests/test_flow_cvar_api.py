from fastapi.testclient import TestClient

from src.api.main import create_app


def test_flow_cvar_reference_endpoint():
    r = TestClient(create_app()).get("/v1/flow-cvar/reference")
    assert r.status_code == 200
    p = r.json()
    assert p["gate"] == "AUTHORIZED"
    assert p["human_review_required"] is True
    assert p["decision_id"].startswith("FLOW-")


def test_flow_cvar_decision_endpoint_preserves_controls():
    r = TestClient(create_app()).post(
        "/v1/flow-cvar/decision?cvar_alpha=0.9&cvar_weight=2&severe_demand=22"
    )
    assert r.status_code == 200
    p = r.json()
    assert p["parameters"] == {"cvar_alpha": 0.9, "cvar_weight": 2.0, "severe_demand": 22.0}
