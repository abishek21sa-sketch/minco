from __future__ import annotations
import os

from src.decision_math.flow_cvar_decision import build_flow_cvar_decision

PROJECT = "MINCO — Hospital Capacity Operations Workstation"
ALGORITHM = "FLOW-CVaR"
SUBTITLE = "Markov patient-flow forecasting, surge capacity, flex staffing, transfer/diversion recourse, and CVaR under correlated demand."
PORT = int(os.getenv("PORT", "8814"))
THEME = "minco"
CONTROLS = [
    {
        "key": "cvar_alpha",
        "label": "CVaR confidence",
        "default": 0.80,
        "min": 0.60,
        "max": 0.95,
        "step": 0.05,
    },
    {
        "key": "cvar_weight",
        "label": "Tail-risk weight",
        "default": 1.5,
        "min": 0,
        "max": 5,
        "step": 0.25,
    },
    {
        "key": "severe_demand",
        "label": "Severe-surge H1 demand",
        "default": 22,
        "min": 15,
        "max": 35,
        "step": 1,
    },
]
DEFAULTS = {x["key"]: x["default"] for x in CONTROLS}
DEMO_STRESS = {"cvar_alpha": 0.90, "cvar_weight": 2.5, "severe_demand": 28}


def _fmt(v):
    if v is None:
        return "N/A"
    return f"{v:,.4f}" if isinstance(v, float) else str(v)


def compute(params):
    raw = build_flow_cvar_decision(
        cvar_alpha=float(params["cvar_alpha"]),
        cvar_weight=float(params["cvar_weight"]),
        severe_demand=float(params["severe_demand"]),
    )
    d = raw["decision"]
    b = raw.get("no_cvar_baseline", {})
    metrics = [
        ["CVaR loss", _fmt(d["cvar_loss"])],
        ["Expected recourse", _fmt(d["expected_recourse_loss"])],
        ["Surge activations", str(d["surge_activations"])],
        ["Flex staff blocks", str(d["flex_staff_blocks"])],
    ]
    actions = []
    for a in raw.get("actions", []):
        actions.append({k.replace("_", " ").title(): v for k, v in a.items()})
    for a in raw.get("scenario_actions", []):
        actions.append({k.replace("_", " ").title(): v for k, v in a.items()})
    baselines = [
        ["FLOW-CVaR tail", _fmt(d["cvar_loss"])],
        ["No-CVaR tail", _fmt(b.get("cvar_loss"))],
        ["Expected transfers", _fmt(d["expected_transfers"])],
        ["Expected deferrals", _fmt(d["expected_deferrals"])],
    ]
    gate = "AUTHORIZED" if all(raw.get("checks", {}).values()) else "BLOCKED"
    return {
        "gate": gate,
        "decision_id": raw.get("decision_id", "FLOW-RUNTIME"),
        "metrics": metrics,
        "actions": actions,
        "baselines": baselines,
        "claim": d["bounded_claim"] + " " + raw.get("operator_note", ""),
        "raw": raw,
    }
