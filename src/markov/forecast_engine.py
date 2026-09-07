from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config.dataclasses import HealthcareInstance
from src.markov.transition_utils import (
    get_state_index_map,
    load_transition_matrices,
)


def _get_sorted_days(arrivals_df: pd.DataFrame) -> List[int]:
    """Return sorted planning days from arrivals data."""
    days = sorted(arrivals_df["day"].unique().tolist())
    return [int(day) for day in days]


def _get_hospitals(instance: HealthcareInstance) -> List[str]:
    """Return sorted hospital IDs."""
    hospitals = sorted(instance.hospitals.df["hospital_id"].unique().tolist())
    return [str(h) for h in hospitals]


def _get_cohorts(instance: HealthcareInstance) -> List[str]:
    """Return sorted cohort IDs from transition bundle."""
    cohorts = sorted(instance.transitions.matrices.keys())
    return [str(c) for c in cohorts]


def _build_arrival_lookup(arrivals_df: pd.DataFrame) -> Dict[Tuple[int, str, str], float]:
    """
    Build lookup for arrivals:
        (day, hospital_id, cohort) -> arrivals
    """
    lookup: Dict[Tuple[int, str, str], float] = {}

    grouped = (
        arrivals_df.groupby(["day", "hospital_id", "cohort"], as_index=False)["arrivals"]
        .sum()
    )

    for row in grouped.itertuples(index=False):
        lookup[(int(row.day), str(row.hospital_id), str(row.cohort))] = float(row.arrivals)

    return lookup


def _zero_state_vector(n_states: int) -> np.ndarray:
    """Return a zero row vector of state counts."""
    return np.zeros(n_states, dtype=float)


def _initialize_state_vectors(
    hospitals: List[str],
    cohorts: List[str],
    n_states: int,
    initial_state: Dict[Tuple[str, str], np.ndarray] | None = None,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Initialize state vectors for each (hospital, cohort).

    Parameters
    ----------
    initial_state:
        Optional mapping (hospital_id, cohort) -> state vector.
        If omitted, all states start at zero.
    """
    state_vectors: Dict[Tuple[str, str], np.ndarray] = {}

    for hospital in hospitals:
        for cohort in cohorts:
            key = (hospital, cohort)
            if initial_state is not None and key in initial_state:
                vec = np.asarray(initial_state[key], dtype=float).copy()
                state_vectors[key] = vec
            else:
                state_vectors[key] = _zero_state_vector(n_states)

    return state_vectors


def _inject_arrivals_into_ed(
    state_vector: np.ndarray,
    arrivals: float,
    ed_index: int,
) -> np.ndarray:
    """
    Add new arrivals into the ED state of a state vector.
    """
    updated = state_vector.copy()
    updated[ed_index] += arrivals
    return updated


def forecast_cohort_state_trajectories(
    instance: HealthcareInstance,
    initial_state: Dict[Tuple[str, str], np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Forecast expected cohort-state counts over the planning horizon.

    Dynamics:
        x_{t+1} = (x_t + arrivals_t_into_ED) P

    Parameters
    ----------
    instance:
        Loaded healthcare instance.
    initial_state:
        Optional mapping (hospital_id, cohort) -> state vector in canonical state order.

    Returns
    -------
    pd.DataFrame
        Columns:
            day, hospital_id, cohort, state, expected_count
        where 'day' is the post-transition state count for that day.
    """
    arrivals_df = instance.arrivals.df.copy()
    transition_matrices = load_transition_matrices(instance.transitions)

    hospitals = _get_hospitals(instance)
    cohorts = _get_cohorts(instance)
    days = _get_sorted_days(arrivals_df)

    state_to_idx = get_state_index_map()
    ed_index = state_to_idx["ED"]
    n_states = len(state_to_idx)

    arrival_lookup = _build_arrival_lookup(arrivals_df)
    state_vectors = _initialize_state_vectors(
        hospitals=hospitals,
        cohorts=cohorts,
        n_states=n_states,
        initial_state=initial_state,
    )

    records = []

    for day in days:
        next_state_vectors: Dict[Tuple[str, str], np.ndarray] = {}

        for hospital in hospitals:
            for cohort in cohorts:
                key = (hospital, cohort)
                current_state = state_vectors[key]
                arrivals = arrival_lookup.get((day, hospital, cohort), 0.0)

                with_arrivals = _inject_arrivals_into_ed(
                    state_vector=current_state,
                    arrivals=arrivals,
                    ed_index=ed_index,
                )

                transition_matrix = transition_matrices[cohort]
                next_state = with_arrivals @ transition_matrix
                next_state_vectors[key] = next_state

                for state_name, idx in state_to_idx.items():
                    records.append(
                        {
                            "day": day,
                            "hospital_id": hospital,
                            "cohort": cohort,
                            "state": state_name,
                            "expected_count": float(next_state[idx]),
                        }
                    )

        state_vectors = next_state_vectors

    return pd.DataFrame(records)


def summarize_total_state_occupancy(
    state_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate expected counts across cohorts by day, hospital, and state.
    """
    summary = (
        state_forecast_df.groupby(["day", "hospital_id", "state"], as_index=False)["expected_count"]
        .sum()
        .sort_values(["day", "hospital_id", "state"])
        .reset_index(drop=True)
    )
    return summary


def summarize_resource_demand(
    state_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert state forecasts into resource demand forecasts.

    Mapping:
    - ICU state -> ICU demand
    - Ward state -> Ward demand
    """
    filtered = state_forecast_df[
        state_forecast_df["state"].isin(["ICU", "Ward"])
    ].copy()
    filtered["resource"] = filtered["state"]

    summary = (
        filtered.groupby(["day", "hospital_id", "resource"], as_index=False)["expected_count"]
        .sum()
        .rename(columns={"expected_count": "expected_demand"})
        .sort_values(["day", "hospital_id", "resource"])
        .reset_index(drop=True)
    )
    return summary


def summarize_network_resource_demand(
    resource_demand_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate resource demand across all hospitals by day and resource.
    """
    summary = (
        resource_demand_df.groupby(["day", "resource"], as_index=False)["expected_demand"]
        .sum()
        .sort_values(["day", "resource"])
        .reset_index(drop=True)
    )
    return summary


def summarize_discharged_census(
    state_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate expected discharged-state census by day and hospital.

    Note:
        This is the expected number of patients in the absorbing Discharged state
        by each day, not the number newly discharged on that day.
    """
    discharged = state_forecast_df[state_forecast_df["state"] == "Discharged"].copy()

    summary = (
        discharged.groupby(["day", "hospital_id"], as_index=False)["expected_count"]
        .sum()
        .rename(columns={"expected_count": "expected_discharged_census"})
        .sort_values(["day", "hospital_id"])
        .reset_index(drop=True)
    )
    return summary


def summarize_daily_new_discharges(
    state_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute expected newly discharged patients by day and hospital.

    This is obtained by differencing the discharged-state census over time.
    Since Discharged is an absorbing state, the daily increment represents
    new discharges on that day.
    """
    discharged_census = summarize_discharged_census(state_forecast_df).copy()
    discharged_census = discharged_census.sort_values(
        ["hospital_id", "day"]
    ).reset_index(drop=True)

    discharged_census["expected_new_discharges"] = (
        discharged_census.groupby("hospital_id")["expected_discharged_census"].diff()
    )

    first_day_mask = discharged_census.groupby("hospital_id").cumcount() == 0
    discharged_census.loc[first_day_mask, "expected_new_discharges"] = (
        discharged_census.loc[first_day_mask, "expected_discharged_census"]
    )

    result = discharged_census[["day", "hospital_id", "expected_new_discharges"]].copy()
    return result


def run_forecast(
    instance: HealthcareInstance,
    initial_state: Dict[Tuple[str, str], np.ndarray] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full expected-value forecast pipeline.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys:
            state_trajectories
            total_state_occupancy
            resource_demand
            network_resource_demand
            discharged_census
            daily_new_discharges
    """
    state_trajectories = forecast_cohort_state_trajectories(
        instance=instance,
        initial_state=initial_state,
    )

    total_state_occupancy = summarize_total_state_occupancy(state_trajectories)
    resource_demand = summarize_resource_demand(state_trajectories)
    network_resource_demand = summarize_network_resource_demand(resource_demand)
    discharged_census = summarize_discharged_census(state_trajectories)
    daily_new_discharges = summarize_daily_new_discharges(state_trajectories)

    return {
        "state_trajectories": state_trajectories,
        "total_state_occupancy": total_state_occupancy,
        "resource_demand": resource_demand,
        "network_resource_demand": network_resource_demand,
        "discharged_census": discharged_census,
        "daily_new_discharges": daily_new_discharges,
    }