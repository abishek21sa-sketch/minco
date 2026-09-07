"""Common-random-number Monte Carlo laboratory for hospital capacity policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .ctmc import CTMCModel, make_reference_patient_flow_ctmc, simulate_population_counts
from .mmpp import simulate_mmpp


@dataclass(frozen=True)
class CapacityPolicy:
    name: str
    icu_surge_beds: int = 0
    ward_surge_beds: int = 0
    flex_staff_beds: int = 0
    transfer_diversion_fraction: float = 0.0
    elective_reduction_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.icu_surge_beds < 0 or self.ward_surge_beds < 0 or self.flex_staff_beds < 0:
            raise ValueError("Capacity additions must be nonnegative")
        for value in (self.transfer_diversion_fraction, self.elective_reduction_fraction):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Policy fractions must be in [0, 1]")


@dataclass(frozen=True)
class MonteCarloConfig:
    horizon_hours: int = 72
    base_icu_beds: int = 24
    base_ward_beds: int = 84
    n_scenarios: int = 1000
    seed: int = 20260817
    cvar_alpha: float = 0.95


def _tail_cvar(values: np.ndarray, alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    values = np.asarray(values, dtype=float)
    threshold = float(np.quantile(values, alpha, method="higher"))
    tail = values[values >= threshold]
    return float(tail.mean()) if tail.size else threshold


def evaluate_policies_common_random_numbers(
    policies: Iterable[CapacityPolicy],
    *,
    transition_matrix: np.ndarray,
    regime_rates: np.ndarray,
    config: MonteCarloConfig = MonteCarloConfig(),
    patient_flow_model: CTMCModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate policies on identical stochastic futures.

    Each scenario gets a deterministic seed derived only from the global seed
    and scenario index. Policy evaluation reuses that same sampled future,
    enabling lower-variance paired comparisons.
    """
    policies = tuple(policies)
    if not policies:
        raise ValueError("At least one policy is required")
    model = patient_flow_model or make_reference_patient_flow_ctmc()
    icu_idx = model.states.index("ICU")
    ward_idx = model.states.index("Ward")

    scenario_rows: list[dict[str, float | int | str]] = []
    for scenario in range(config.n_scenarios):
        scenario_seed = int(config.seed + 1009 * scenario)
        states, arrivals = simulate_mmpp(
            transition_matrix=transition_matrix,
            rates=regime_rates,
            n_periods=config.horizon_hours,
            seed=scenario_seed,
        )
        rng = np.random.default_rng(scenario_seed + 17)
        capacity_shock = float(rng.choice([1.0, 0.95, 0.90], p=[0.86, 0.10, 0.04]))
        staff_absence = float(np.clip(rng.beta(2.0, 35.0), 0.0, 0.20))

        # Same underlying arrivals/pathway randomness for every policy. Policies
        # differ only in deterministic intervention transformations below.
        for policy in policies:
            adjusted_arrivals = arrivals.astype(float).copy()
            # Transfer diversion and elective reduction remove only the portion
            # of arrivals explicitly controllable by the policy.
            controllable_fraction = 0.22
            reduction = controllable_fraction * (
                0.55 * policy.transfer_diversion_fraction
                + 0.45 * policy.elective_reduction_fraction
            )
            adjusted_arrivals = np.rint(adjusted_arrivals * (1.0 - reduction)).astype(int)

            trajectory = simulate_population_counts(
                model,
                initial_counts=[0] * len(model.states),
                arrivals_by_step=adjusted_arrivals,
                step_hours=1.0,
                seed=scenario_seed + 41,
            )
            icu_capacity = max(
                1.0,
                (config.base_icu_beds + policy.icu_surge_beds + policy.flex_staff_beds)
                * capacity_shock
                * (1.0 - staff_absence),
            )
            ward_capacity = max(
                1.0,
                (config.base_ward_beds + policy.ward_surge_beds + policy.flex_staff_beds)
                * capacity_shock
                * (1.0 - staff_absence),
            )
            icu_overload = np.maximum(trajectory[1:, icu_idx] - icu_capacity, 0.0)
            ward_overload = np.maximum(trajectory[1:, ward_idx] - ward_capacity, 0.0)
            max_icu_util = float(np.max(trajectory[1:, icu_idx] / icu_capacity))
            max_ward_util = float(np.max(trajectory[1:, ward_idx] / ward_capacity))
            loss = float(3.0 * icu_overload.sum() + 1.2 * ward_overload.sum())
            scenario_rows.append({
                "scenario_id": scenario,
                "policy": policy.name,
                "scenario_seed": scenario_seed,
                "peak_regime": int(np.max(states)),
                "total_arrivals": int(arrivals.sum()),
                "capacity_shock": capacity_shock,
                "staff_absence_rate": staff_absence,
                "max_icu_utilization": max_icu_util,
                "max_ward_utilization": max_ward_util,
                "icu_overload_bed_hours": float(icu_overload.sum()),
                "ward_overload_bed_hours": float(ward_overload.sum()),
                "loss": loss,
            })

    detail = pd.DataFrame(scenario_rows)
    summaries: list[dict[str, float | str | int]] = []
    for policy_name, group in detail.groupby("policy", sort=True):
        losses = group["loss"].to_numpy(dtype=float)
        summaries.append({
            "policy": policy_name,
            "n_scenarios": int(len(group)),
            "expected_loss": float(losses.mean()),
            "p95_loss": float(np.quantile(losses, 0.95)),
            "cvar_loss": _tail_cvar(losses, config.cvar_alpha),
            "probability_icu_overload": float((group["icu_overload_bed_hours"] > 0).mean()),
            "probability_ward_overload": float((group["ward_overload_bed_hours"] > 0).mean()),
            "mean_peak_icu_utilization": float(group["max_icu_utilization"].mean()),
        })
    return detail, pd.DataFrame(summaries).sort_values("expected_loss").reset_index(drop=True)


def paired_policy_difference(
    detail: pd.DataFrame,
    *,
    baseline_policy: str,
    candidate_policy: str,
    metric: str = "loss",
) -> np.ndarray:
    """Return candidate-minus-baseline scenario-paired differences."""
    pivot = detail.pivot(index="scenario_id", columns="policy", values=metric)
    if baseline_policy not in pivot or candidate_policy not in pivot:
        raise ValueError("Both policies must exist in the detail frame")
    return (pivot[candidate_policy] - pivot[baseline_policy]).to_numpy(dtype=float)
