"""FLOW-CVaR: flow-conserving tail-risk capacity planning for MINCO.

This module is a governed signature layer over MINCO's independently validated
stochastic capacity MILP.  It does not replace the Julia/JuMP/Gurobi engine;
it exposes the patient-flow and CVaR semantics as a compact, auditable decision
artifact that can also be validated with SciPy/HiGHS on small instances.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .stochastic_milp_oracle import (
    OracleSolution,
    StochasticCapacityInstance,
    solve_small_stochastic_milp,
    validate_stochastic_solution,
)


@dataclass(frozen=True)
class FlowCVaRDecision:
    algorithm: str
    status: str
    objective: float
    expected_recourse_loss: float
    cvar_loss: float
    cvar_alpha: float
    cvar_weight: float
    surge_activations: int
    flex_staff_blocks: int
    expected_unsafe_excess: float
    expected_boarding: float
    expected_deferrals: float
    expected_transfers: float
    tail_scenario_loss: float
    feasible: bool
    bounded_claim: str


def _weighted_cvar(losses: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    """Discrete CVaR using the same Rockafellar-Uryasev representation as the MILP."""
    candidates = np.unique(np.asarray(losses, dtype=float))
    return float(min(
        eta + np.dot(probabilities, np.maximum(losses - eta, 0.0)) / (1.0 - alpha)
        for eta in candidates
    ))


def summarize_flow_cvar(inst: StochasticCapacityInstance, solution: OracleSolution) -> FlowCVaRDecision:
    inst.validate()
    if solution.status != "OPTIMAL":
        return FlowCVaRDecision(
            algorithm="FLOW-CVaR", status=solution.status, objective=float(solution.objective),
            expected_recourse_loss=float("nan"), cvar_loss=float("nan"),
            cvar_alpha=inst.cvar_alpha, cvar_weight=inst.cvar_weight,
            surge_activations=0, flex_staff_blocks=0,
            expected_unsafe_excess=float("nan"), expected_boarding=float("nan"),
            expected_deferrals=float("nan"), expected_transfers=float("nan"),
            tail_scenario_loss=float("nan"), feasible=False,
            bounded_claim="No operational claim: optimizer did not return an optimal solution.",
        )
    probs = np.asarray(inst.scenario_probabilities, dtype=float)
    validation = validate_stochastic_solution(inst, solution)
    expected_loss = float(np.dot(probs, solution.scenario_loss))
    cvar = _weighted_cvar(solution.scenario_loss, probs, inst.cvar_alpha)
    # recourse arrays are [scenario,hospital,period] except transfer [scenario,h,h,period]
    exp_unsafe = float(np.dot(probs, solution.unsafe.sum(axis=(1, 2))))
    exp_boarding = float(np.dot(probs, solution.boarding.sum(axis=(1, 2))))
    exp_deferrals = float(np.dot(probs, solution.defer.sum(axis=(1, 2))))
    exp_transfers = float(np.dot(probs, solution.transfer.sum(axis=(1, 2, 3))))
    return FlowCVaRDecision(
        algorithm="FLOW-CVaR",
        status=solution.status,
        objective=float(solution.objective),
        expected_recourse_loss=expected_loss,
        cvar_loss=cvar,
        cvar_alpha=float(inst.cvar_alpha),
        cvar_weight=float(inst.cvar_weight),
        surge_activations=int(solution.surge.sum()),
        flex_staff_blocks=int(solution.flex_blocks.sum()),
        expected_unsafe_excess=exp_unsafe,
        expected_boarding=exp_boarding,
        expected_deferrals=exp_deferrals,
        expected_transfers=exp_transfers,
        tail_scenario_loss=float(np.max(solution.scenario_loss)),
        feasible=bool(validation.get("feasible", False)),
        bounded_claim=(
            "FLOW-CVaR is a model-based capacity decision under the supplied demand scenarios, "
            "costs, capacities, transfer topology, and CVaR settings; it is not a clinical outcome guarantee."
        ),
    )


def solve_flow_cvar(inst: StochasticCapacityInstance) -> FlowCVaRDecision:
    return summarize_flow_cvar(inst, solve_small_stochastic_milp(inst))


def cvar_ablation(inst: StochasticCapacityInstance) -> dict[str, FlowCVaRDecision]:
    """Compare the governed tail-risk plan with an expected-cost-only ablation."""
    governed = solve_flow_cvar(inst)
    no_tail = solve_flow_cvar(replace(inst, cvar_weight=0.0))
    return {"flow_cvar": governed, "no_cvar": no_tail}


def as_dict(decision: FlowCVaRDecision) -> dict[str, Any]:
    return {name: getattr(decision, name) for name in decision.__dataclass_fields__}
