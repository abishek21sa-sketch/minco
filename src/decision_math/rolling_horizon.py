"""Python/HiGHS rolling-horizon reference used to validate decision semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .stochastic_milp_oracle import StochasticCapacityInstance, solve_small_stochastic_milp


@dataclass(frozen=True)
class RollingHorizonDecision:
    epoch: int
    status: str
    surge_first_period: np.ndarray
    flex_first_period: np.ndarray
    objective: float


def rolling_horizon_reference(
    epochs: Iterable[int],
    instance_factory: Callable[[int], StochasticCapacityInstance],
) -> list[RollingHorizonDecision]:
    decisions: list[RollingHorizonDecision] = []
    for epoch in epochs:
        solution = solve_small_stochastic_milp(instance_factory(int(epoch)))
        if solution.status != "OPTIMAL":
            raise RuntimeError(f"Rolling-horizon oracle failed at epoch {epoch}: {solution.status}")
        decisions.append(RollingHorizonDecision(
            epoch=int(epoch),
            status=solution.status,
            surge_first_period=solution.surge[:, 0].copy(),
            flex_first_period=solution.flex_blocks[:, 0].copy(),
            objective=float(solution.objective),
        ))
    return decisions
