from __future__ import annotations

# Path bootstrap is intentional for standalone evidence execution.
# ruff: noqa: I001

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decision_math.flow_cvar import summarize_flow_cvar  # noqa: E402
from src.decision_math.stochastic_milp_oracle import solve_small_stochastic_milp  # noqa: E402
from src.minco4x.signature_algorithm import make_network_flow_instance, solve_flow_cvar_network  # noqa: E402


def weighted_cvar(loss, p, alpha):
    return float(min(e + np.dot(p, np.maximum(loss - e, 0)) / (1 - alpha) for e in np.unique(loss)))


def greedy_diversion(inst):
    losses = []
    for wi in range(len(inst.scenario_probabilities)):
        d = inst.demand[wi, :, 0].astype(float).copy()
        cap = inst.base_beds.astype(float).copy()
        transfers = 0.0
        deferred = 0.0
        for i, j in [(0, 1), (1, 0)]:
            excess = max(0.0, d[i] - cap[i])
            spare = max(0.0, cap[j] - d[j])
            q = min(excess, spare, float(inst.transfer_capacity[i, j, 0]))
            d[i] -= q
            d[j] += q
            transfers += q
        for h in range(len(inst.hospitals)):
            q = min(max(0.0, d[h] - cap[h]), float(inst.elective_deferral_limit[h, 0]))
            d[h] -= q
            deferred += q
        unsafe = sum(
            max(0.0, d[h] - inst.safe_fraction[h] * cap[h]) for h in range(len(inst.hospitals))
        )
        boarding = sum(max(0.0, d[h] - cap[h]) for h in range(len(inst.hospitals)))
        losses.append(
            inst.transfer_cost * transfers
            + inst.defer_cost * deferred
            + inst.unsafe_cost * unsafe
            + inst.boarding_cost * boarding
        )
    loss = np.asarray(losses)
    p = np.asarray(inst.scenario_probabilities)
    return {
        "expected_recourse_loss": float(p @ loss),
        "cvar_loss": weighted_cvar(loss, p, inst.cvar_alpha),
        "tail_scenario_loss": float(loss.max()),
    }


def main():
    out = ROOT / "artifacts" / "flow_cvar"
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    canonical = solve_flow_cvar_network()
    runtime = time.perf_counter() - t
    inst, flow = make_network_flow_instance()
    noflex_sol = solve_small_stochastic_milp(
        inst, fixed_surge=np.zeros((2, 1)), fixed_flex=np.zeros((2, 1))
    )
    noflex = summarize_flow_cvar(inst, noflex_sol)
    mean_demand = np.tensordot(inst.scenario_probabilities, inst.demand, axes=(0, 0))[None, :, :]
    deterministic = replace(
        inst, scenario_probabilities=np.array([1.0]), demand=mean_demand, cvar_weight=0.0
    )
    det_sol = solve_small_stochastic_milp(deterministic)
    det_eval = summarize_flow_cvar(
        inst,
        solve_small_stochastic_milp(
            inst, fixed_surge=det_sol.surge, fixed_flex=det_sol.flex_blocks
        ),
    )
    greedy = greedy_diversion(inst)
    sens = []
    for alpha in (0.70, 0.80, 0.90):
        for risk in (0.0, 1.5, 3.0):
            for severe in (18.0, 22.0):
                r = solve_flow_cvar_network(
                    cvar_alpha=alpha, cvar_weight=risk, severe_demand=severe
                )
                sens.append(
                    {
                        "alpha": alpha,
                        "cvar_weight": risk,
                        "severe_demand": severe,
                        "status": r["decision"]["status"],
                        "cvar_loss": r["decision"]["cvar_loss"],
                        "expected_loss": r["decision"]["expected_recourse_loss"],
                        "transfers": r["raw_transfer_total"],
                        "deferrals": r["raw_deferral_total"],
                    }
                )
    d = canonical["decision"]
    checks = {
        "canonical_optimal": d["status"] == "OPTIMAL",
        "canonical_feasible": bool(d["feasible"]),
        "markov_rows_stochastic": bool(np.allclose(flow["row_sums"], 1.0)),
        "two_hospital_network": len(flow["forecast_hospital_census"]) == 2,
        "transfer_and_diversion_exercised": canonical["raw_transfer_total"] > 0
        and canonical["raw_deferral_total"] > 0,
        "beats_no_flex_tail": d["cvar_loss"] <= noflex.cvar_loss + 1e-9,
        "beats_deterministic_expected_demand_tail": d["cvar_loss"] <= det_eval.cvar_loss + 1e-9,
        "beats_greedy_diversion_tail": d["cvar_loss"] <= greedy["cvar_loss"] + 1e-9,
        "sensitivity_complete": len(sens) == 18 and all(r["status"] == "OPTIMAL" for r in sens),
    }
    payload = {
        "algorithm": "FLOW-CVaR",
        "null_hypothesis": "Under common patient-flow forecasts and surge scenarios, FLOW-CVaR does not reduce tail recourse loss relative to no-flex, deterministic expected-demand, or greedy-diversion policies.",
        "evidence_class": "synthetic algorithmic validation",
        "runtime_seconds": runtime,
        "seed": None,
        "markov_flow": flow,
        "canonical": canonical,
        "baselines": {
            "no_flex": asdict(noflex),
            "deterministic_expected_demand": asdict(det_eval),
            "greedy_diversion": greedy,
        },
        "checks": checks,
        "claim_boundary": d["bounded_claim"],
    }
    (out / "signature_evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(sens).to_csv(out / "signature_sensitivity.csv", index=False)
    passed = sum(checks.values())
    print(f"FLOW_CVAR_SIGNATURE_EVIDENCE={passed}/{len(checks)}")
    print(f"FLOW_CVAR_SIGNATURE_SENSITIVITY={len(sens)}")
    print(f"FLOW_CVAR_TAIL={d['cvar_loss']:.3f}")
    print(f"NO_FLEX_TAIL={noflex.cvar_loss:.3f}")
    print(f"DETERMINISTIC_TAIL={det_eval.cvar_loss:.3f}")
    print(f"GREEDY_TAIL={greedy['cvar_loss']:.3f}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
