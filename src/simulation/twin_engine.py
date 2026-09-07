from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.dataclasses import HealthcareInstance, STATE_ORDER
from src.simulation.arrival_generator import generate_arrival_scenario
from src.simulation.transition_sampler import (
    build_transition_matrix_lookup,
    sample_one_step_with_arrivals,
)


StateKey = Tuple[str, str]  # (hospital_id, cohort)


def _get_hospitals(instance: HealthcareInstance) -> list[str]:
    hospitals = sorted(instance.hospitals.df["hospital_id"].unique().tolist())
    return [str(h) for h in hospitals]


def _get_cohorts(instance: HealthcareInstance) -> list[str]:
    cohorts = sorted(instance.transitions.matrices.keys())
    return [str(c) for c in cohorts]


def _get_days_from_realized_arrivals(realized_arrivals_df: pd.DataFrame) -> list[int]:
    days = sorted(realized_arrivals_df["day"].unique().tolist())
    return [int(day) for day in days]


def _build_realized_arrival_lookup(
    realized_arrivals_df: pd.DataFrame,
) -> Dict[Tuple[int, str, str], int]:
    """
    Build lookup:
        (day, hospital_id, cohort) -> realized_arrivals
    """
    grouped = (
        realized_arrivals_df.groupby(
            ["day", "hospital_id", "cohort"], as_index=False
        )["realized_arrivals"]
        .sum()
    )

    lookup: Dict[Tuple[int, str, str], int] = {}
    for row in grouped.itertuples(index=False):
        lookup[(int(row.day), str(row.hospital_id), str(row.cohort))] = int(
            row.realized_arrivals
        )

    return lookup


def _initialize_state_counts(
    hospitals: list[str],
    cohorts: list[str],
    initial_state: Optional[Dict[StateKey, np.ndarray]] = None,
) -> Dict[StateKey, np.ndarray]:
    """
    Initialize state-count vectors for each (hospital, cohort).
    """
    n_states = len(STATE_ORDER)
    state_counts: Dict[StateKey, np.ndarray] = {}

    for hospital in hospitals:
        for cohort in cohorts:
            key = (hospital, cohort)
            if initial_state is not None and key in initial_state:
                state_counts[key] = np.asarray(initial_state[key], dtype=int).copy()
            else:
                state_counts[key] = np.zeros(n_states, dtype=int)

    return state_counts


def simulate_stochastic_twin(
    instance: HealthcareInstance,
    realized_arrivals_df: Optional[pd.DataFrame] = None,
    initial_state: Optional[Dict[StateKey, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Simulate a stochastic hospital-network twin using realized arrivals
    and stochastic cohort transitions.

    Parameters
    ----------
    instance:
        Loaded healthcare instance.
    realized_arrivals_df:
        Optional realized arrivals scenario. If omitted, one is generated.
        Expected columns:
            day, hospital_id, cohort, arrivals, realized_arrivals
    initial_state:
        Optional mapping:
            (hospital_id, cohort) -> integer state-count vector
    rng:
        Optional NumPy random generator.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys:
            realized_arrivals
            state_trajectories
            total_state_occupancy
            resource_demand
            network_resource_demand
    """
    if rng is None:
        rng = np.random.default_rng()

    if realized_arrivals_df is None:
        realized_arrivals_df = generate_arrival_scenario(instance=instance, rng=rng)

    hospitals = _get_hospitals(instance)
    cohorts = _get_cohorts(instance)
    days = _get_days_from_realized_arrivals(realized_arrivals_df)

    transition_lookup = build_transition_matrix_lookup(instance)
    arrival_lookup = _build_realized_arrival_lookup(realized_arrivals_df)
    state_counts = _initialize_state_counts(
        hospitals=hospitals,
        cohorts=cohorts,
        initial_state=initial_state,
    )

    records = []

    for day in days:
        next_state_counts: Dict[StateKey, np.ndarray] = {}

        for hospital in hospitals:
            for cohort in cohorts:
                key = (hospital, cohort)
                current_counts = state_counts[key]
                arrivals = arrival_lookup.get((day, hospital, cohort), 0)

                next_counts = sample_one_step_with_arrivals(
                    current_counts=current_counts,
                    arrivals=arrivals,
                    transition_matrix=transition_lookup[cohort],
                    rng=rng,
                )
                next_state_counts[key] = next_counts

                for state_name, count in zip(STATE_ORDER, next_counts):
                    records.append(
                        {
                            "day": int(day),
                            "hospital_id": hospital,
                            "cohort": cohort,
                            "state": state_name,
                            "count": int(count),
                        }
                    )

        state_counts = next_state_counts

    state_trajectories = pd.DataFrame(records)

    total_state_occupancy = (
        state_trajectories.groupby(
            ["day", "hospital_id", "state"], as_index=False
        )["count"]
        .sum()
        .sort_values(["day", "hospital_id", "state"])
        .reset_index(drop=True)
    )

    resource_demand = (
        total_state_occupancy[total_state_occupancy["state"].isin(["ICU", "Ward"])]
        .copy()
        .rename(columns={"state": "resource", "count": "realized_demand"})
        .sort_values(["day", "hospital_id", "resource"])
        .reset_index(drop=True)
    )

    network_resource_demand = (
        resource_demand.groupby(["day", "resource"], as_index=False)["realized_demand"]
        .sum()
        .sort_values(["day", "resource"])
        .reset_index(drop=True)
    )

    return {
        "realized_arrivals": realized_arrivals_df.copy(),
        "state_trajectories": state_trajectories,
        "total_state_occupancy": total_state_occupancy,
        "resource_demand": resource_demand,
        "network_resource_demand": network_resource_demand,
    }