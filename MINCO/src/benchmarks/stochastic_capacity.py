"""Reproducible benchmark generators for the Phase-1 stochastic capacity engine."""
from __future__ import annotations

import numpy as np

from src.decision_math.stochastic_milp_oracle import StochasticCapacityInstance
from src.scenarios.operating_modes import get_operating_mode


def make_network_capacity_benchmark(
    *,
    n_hospitals: int = 3,
    n_periods: int = 4,
    n_scenarios: int = 8,
    seed: int = 20260817,
    operating_mode: str = "normal",
) -> StochasticCapacityInstance:
    if n_hospitals < 1 or n_periods < 1 or n_scenarios < 2:
        raise ValueError("Benchmark requires hospitals>=1, periods>=1, scenarios>=2")
    rng = np.random.default_rng(seed)
    mode = get_operating_mode(operating_mode)
    hospitals = tuple(f"H{i+1}" for i in range(n_hospitals))
    periods = tuple(range(n_periods))
    probabilities = np.full(n_scenarios, 1.0 / n_scenarios)
    base_beds = np.linspace(24, 42, n_hospitals)
    safe_fraction = np.linspace(0.86, 0.91, n_hospitals)
    surge_beds = np.maximum(4.0, np.rint(0.22 * base_beds))
    surge_cost = 5.0 + 0.35 * surge_beds
    flex_beds_per_block = np.maximum(2.0, np.rint(0.10 * base_beds))
    flex_cost = 4.0 + 0.4 * flex_beds_per_block
    max_flex_blocks = np.full(n_hospitals, 3)
    base_demand = (
        base_beds[:, None]
        * np.linspace(0.82, 1.02, n_periods)[None, :]
        * mode.demand_multiplier
    )
    demand = np.empty((n_scenarios, n_hospitals, n_periods), dtype=float)
    for w in range(n_scenarios):
        surge_factor = rng.choice([0.90, 1.0, 1.10, 1.25], p=[0.10, 0.45, 0.30, 0.15])
        mode_noise = 2.0 * mode.ed_multiplier
        demand[w] = np.maximum(0.0, np.rint(base_demand * surge_factor + rng.normal(0, mode_noise, base_demand.shape)))
    deferral = np.maximum(1.0, np.rint(0.16 * base_beds[:, None] * np.ones((1,n_periods))))
    transfer = np.zeros((n_hospitals, n_hospitals, n_periods), dtype=float)
    for i in range(n_hospitals):
        for j in range(n_hospitals):
            if i != j:
                transfer[i,j,:] = max(1.0, round(0.12 * base_beds[i] * mode.transfer_multiplier))
    return StochasticCapacityInstance(
        hospitals=hospitals,
        periods=periods,
        scenario_probabilities=probabilities,
        demand=demand,
        base_beds=base_beds,
        safe_fraction=safe_fraction,
        surge_beds=surge_beds,
        surge_cost=surge_cost,
        flex_beds_per_block=flex_beds_per_block,
        flex_cost=flex_cost,
        max_flex_blocks=max_flex_blocks,
        elective_deferral_limit=deferral,
        transfer_capacity=transfer,
        transfer_cost=0.75,
        defer_cost=1.7,
        unsafe_cost=8.0,
        boarding_cost=24.0,
        cvar_alpha=0.90,
        cvar_weight=0.40,
    )
