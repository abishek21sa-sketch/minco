"""Small-instance Progressive Hedging reference implementation.

This is an independently executable algorithmic check for MINCO's primary
Julia/JuMP Progressive Hedging implementation. It supports the one-hospital,
one-period benchmark where first-stage decisions are surge activation and flex
staff blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .stochastic_milp_oracle import StochasticCapacityInstance


@dataclass(frozen=True)
class ProgressiveHedgingResult:
    consensus_surge: int
    consensus_flex: int
    iterations: int
    converged: bool
    scenario_decisions: np.ndarray
    residual_history: tuple[float, ...]


def _scenario_recourse_loss(inst: StochasticCapacityInstance, wi: int, surge: int, flex: int) -> float:
    if len(inst.hospitals) != 1 or len(inst.periods) != 1:
        raise ValueError("Reference PH supports one hospital and one period")
    physical = inst.base_beds[0] + surge * inst.surge_beds[0] + flex * inst.flex_beds_per_block[0]
    safe = inst.safe_fraction[0] * physical
    demand = float(inst.demand[wi, 0, 0])
    max_def = int(round(float(inst.elective_deferral_limit[0, 0])))
    best = float("inf")
    for defer in range(max_def + 1):
        treated = max(0.0, demand - defer)
        unsafe = max(0.0, treated - safe)
        boarding = max(0.0, treated - physical)
        loss = inst.defer_cost * defer + inst.unsafe_cost * unsafe + inst.boarding_cost * boarding
        best = min(best, float(loss))
    return best


def progressive_hedging_single_hospital(
    inst: StochasticCapacityInstance,
    *,
    rho: float = 12.0,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> ProgressiveHedgingResult:
    inst.validate()
    if rho <= 0 or max_iter <= 0:
        raise ValueError("rho and max_iter must be positive")
    w_count = len(inst.scenario_probabilities)
    candidates = np.array(
        list(product([0, 1], range(int(inst.max_flex_blocks[0]) + 1))),
        dtype=float,
    )
    first_cost = candidates[:, 0] * inst.surge_cost[0] + candidates[:, 1] * inst.flex_cost[0]

    multipliers = np.zeros((w_count, 2), dtype=float)
    decisions = np.zeros((w_count, 2), dtype=float)
    # Initial independent scenario solves.
    for wi in range(w_count):
        objectives = np.array([
            first_cost[k] + _scenario_recourse_loss(inst, wi, int(x[0]), int(x[1]))
            for k, x in enumerate(candidates)
        ])
        decisions[wi] = candidates[int(np.argmin(objectives))]

    residuals: list[float] = []
    converged = False
    for iteration in range(1, max_iter + 1):
        xbar = np.average(decisions, axis=0, weights=inst.scenario_probabilities)
        new_decisions = np.empty_like(decisions)
        for wi in range(w_count):
            values = []
            for k, candidate in enumerate(candidates):
                augmented = (
                    first_cost[k]
                    + _scenario_recourse_loss(inst, wi, int(candidate[0]), int(candidate[1]))
                    + float(multipliers[wi] @ candidate)
                    + 0.5 * rho * float(np.sum((candidate - xbar) ** 2))
                )
                values.append(augmented)
            new_decisions[wi] = candidates[int(np.argmin(values))]
        decisions = new_decisions
        xbar = np.average(decisions, axis=0, weights=inst.scenario_probabilities)
        residual = float(np.sqrt(np.sum(inst.scenario_probabilities[:, None] * (decisions - xbar) ** 2)))
        residuals.append(residual)
        if residual <= tolerance:
            converged = True
            break
        multipliers += rho * (decisions - xbar)

    # PH consensus is integral only when all scenario decisions agree.
    consensus = np.rint(np.average(decisions, axis=0, weights=inst.scenario_probabilities)).astype(int)
    return ProgressiveHedgingResult(
        consensus_surge=int(consensus[0]),
        consensus_flex=int(consensus[1]),
        iterations=iteration,
        converged=converged,
        scenario_decisions=decisions.astype(int),
        residual_history=tuple(residuals),
    )
