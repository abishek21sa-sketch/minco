from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.decision_math.flow_cvar import as_dict, cvar_ablation, solve_flow_cvar
from src.decision_math.stochastic_milp_oracle import make_tiny_oracle_instance

OUT = Path("artifacts/flow_cvar")
OUT.mkdir(parents=True, exist_ok=True)
base = make_tiny_oracle_instance()
rare = replace(
    base,
    scenario_probabilities=np.array([0.9, 0.1]),
    demand=np.array([[[9.0]], [[20.0]]]),
    surge_cost=np.array([18.0]),
    cvar_alpha=0.8,
    cvar_weight=1.5,
)
ab = cvar_ablation(rare)
checks = {
    "canonical_optimal": solve_flow_cvar(base).status == "OPTIMAL",
    "canonical_feasible": solve_flow_cvar(base).feasible,
    "tail_plan_feasible": ab["flow_cvar"].feasible,
    "ablation_feasible": ab["no_cvar"].feasible,
    "cvar_not_worse": ab["flow_cvar"].cvar_loss <= ab["no_cvar"].cvar_loss + 1e-9,
    "tail_term_changes_first_stage": (
        ab["flow_cvar"].surge_activations,
        ab["flow_cvar"].flex_staff_blocks,
    )
    != (ab["no_cvar"].surge_activations, ab["no_cvar"].flex_staff_blocks),
}
rows = []
for weight in [0, 0.25, 0.5, 1, 1.5, 2]:
    for alpha in [0.7, 0.8, 0.9]:
        d = solve_flow_cvar(replace(rare, cvar_weight=weight, cvar_alpha=alpha))
        rows.append({"cvar_weight": weight, "alpha": alpha, **as_dict(d)})
(OUT / "evidence.json").write_text(
    json.dumps({"checks": checks, "ablation": {k: as_dict(v) for k, v in ab.items()}}, indent=2),
    encoding="utf-8",
)
with (OUT / "sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
passed = sum(checks.values())
(OUT / "EVIDENCE_REPORT.md").write_text(
    f"""# FLOW-CVaR Evidence\n\nChecks: **{passed}/{len(checks)} passed**.\n\nThe rare-tail benchmark compares FLOW-CVaR against an expected-cost-only ablation on the same patient-demand scenarios and capacity model. The evidence is algorithmic validation only; it is not clinical validation.\n\n- FLOW-CVaR CVaR loss: {ab["flow_cvar"].cvar_loss:.3f}\n- No-CVaR CVaR loss: {ab["no_cvar"].cvar_loss:.3f}\n- FLOW-CVaR first stage: surge={ab["flow_cvar"].surge_activations}, flex={ab["flow_cvar"].flex_staff_blocks}\n- No-CVaR first stage: surge={ab["no_cvar"].surge_activations}, flex={ab["no_cvar"].flex_staff_blocks}\n""",
    encoding="utf-8",
)
print(f"FLOW_CVAR_EVIDENCE={passed}/{len(checks)}")
print(f"FLOW_CVAR_SENSITIVITY={len(rows)}")
print(f"FLOW_CVAR_TAIL={ab['flow_cvar'].cvar_loss:.3f}")
print(f"NO_CVAR_TAIL={ab['no_cvar'].cvar_loss:.3f}")
