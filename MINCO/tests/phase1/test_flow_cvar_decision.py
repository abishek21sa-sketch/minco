from src.decision_math.flow_cvar_decision import build_flow_cvar_decision


def test_flow_cvar_governed_decision_is_authorized_and_review_gated():
    d = build_flow_cvar_decision()
    assert d["gate"] == "AUTHORIZED"
    assert d["human_review_required"] is True
    assert all(d["checks"].values())
    assert d["decision_id"].startswith("FLOW-")


def test_flow_cvar_decision_id_is_deterministic():
    assert build_flow_cvar_decision()["decision_id"] == build_flow_cvar_decision()["decision_id"]


def test_flow_cvar_reports_no_cvar_counterfactual():
    d = build_flow_cvar_decision()
    assert d["decision"]["cvar_loss"] <= d["no_cvar_baseline"]["cvar_loss"] + 1e-9


def test_flow_cvar_controls_are_preserved():
    d = build_flow_cvar_decision(cvar_alpha=0.9, cvar_weight=2.0, severe_demand=22.0)
    assert d["parameters"] == {"cvar_alpha": 0.9, "cvar_weight": 2.0, "severe_demand": 22.0}
