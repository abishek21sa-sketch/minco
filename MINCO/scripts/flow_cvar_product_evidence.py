from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.decision_math.flow_cvar_decision import build_flow_cvar_decision  # noqa: E402


def main():
    out = ROOT / "artifacts" / "flow_cvar"
    out.mkdir(parents=True, exist_ok=True)
    d = build_flow_cvar_decision()
    checks = {
        "decision_authorized": d["gate"] == "AUTHORIZED",
        "human_review_required": d["human_review_required"] is True,
        "all_evidence_checks_pass": all(d["checks"].values()),
        "actions_present": len(d["actions"]) >= 1,
        "counterfactual_present": "no_cvar_baseline" in d,
        "tail_risk_improved": d["decision"]["cvar_loss"]
        <= d["no_cvar_baseline"]["cvar_loss"] + 1e-9,
        "evidence_class_bounded": d["evidence_class"] == "synthetic algorithmic validation",
        "markov_flow_exposed": "flow_forecast" in d
        and len(d["flow_forecast"]["transition_matrix"]) == 3,
        "network_recourse_exposed": len(d.get("scenario_actions", [])) >= 1,
    }
    payload = {"decision": d, "product_checks": checks}
    (out / "product_decision_evidence.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    passed = sum(checks.values())
    report = [
        "# FLOW-CVaR Product Integration Evidence",
        "",
        f"Checks: **{passed}/{len(checks)} passed**.",
        "",
        f"- Decision ID: `{d['decision_id']}`",
        f"- Gate: **{d['gate']}**",
        f"- Human review required: **{d['human_review_required']}**",
        f"- FLOW-CVaR tail loss: **{d['decision']['cvar_loss']:.3f}**",
        f"- No-CVaR tail loss: **{d['no_cvar_baseline']['cvar_loss']:.3f}**",
        "",
        "## Boundary",
        "",
        d["operator_note"],
        "",
        d["decision"]["bounded_claim"],
    ]
    (out / "PRODUCT_INTEGRATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"FLOW_CVAR_PRODUCT_EVIDENCE={passed}/{len(checks)}")
    print(f"FLOW_CVAR_DECISION_ID={d['decision_id']}")
    print(f"FLOW_CVAR_GATE={d['gate']}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
