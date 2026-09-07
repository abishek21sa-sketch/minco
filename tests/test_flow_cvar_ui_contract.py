from pathlib import Path

APP = Path("dashboard/app.py").read_text(encoding="utf-8")


def test_flow_cvar_operator_surface_registered():
    assert '"FLOW-CVaR Council"' in APP and "render_flow_cvar_tab" in APP and "with tab9:" in APP


def test_flow_cvar_surface_exposes_governance_and_counterfactuals():
    for token in [
        "CVaR confidence",
        "Tail-risk weight",
        "Rare-surge H1 demand",
        "Markov patient-flow forecast",
        "Scenario transfer / diversion actions",
        "Evidence gate",
        "Tail-risk counterfactual",
        "Operator approval is mandatory",
    ]:
        assert token in APP
