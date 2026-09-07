"""Healthcare-flow Industrial Engineering mathematics for MINCO.

The module translates patient-arrival forecasts into resource consequences by
combining absorbing-Markov expected residence time with Little's Law.

For a transient Markov submatrix Q, the fundamental matrix is

    N = (I - Q)^(-1)

and N[i, j] is the expected number of time steps spent in transient state j
when starting in transient state i.  MINCO's reference transition step is one
day, so the ICU and Ward entries are expected bed-days per arrival.

For a steady arrival rate lambda (patients/day), Little's Law gives

    L = lambda * W

where W is expected resource-days per patient and L is expected census (beds).
This is an engineering approximation used to translate the AI demand forecast
into capacity pressure.  It is not a clinical prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.dataclasses import HealthcareInstance
from src.markov.transition_utils import get_state_index_map, load_transition_matrices

TRANSIENT_STATES: tuple[str, ...] = ("ED", "ICU", "Ward")
RESOURCE_STATES: tuple[str, ...] = ("ICU", "Ward")


def expected_transient_visits(
    transition_matrix: np.ndarray,
    *,
    start_state: str = "ED",
) -> dict[str, float]:
    """Compute expected transient-state visits using the fundamental matrix."""
    state_index = get_state_index_map()
    transient_indices = [state_index[state] for state in TRANSIENT_STATES]
    if transition_matrix.shape != (len(state_index), len(state_index)):
        raise ValueError(
            f"Expected transition matrix shape {(len(state_index), len(state_index))}, "
            f"got {transition_matrix.shape}"
        )
    if start_state not in TRANSIENT_STATES:
        raise ValueError(f"start_state must be one of {TRANSIENT_STATES}")

    q = transition_matrix[np.ix_(transient_indices, transient_indices)]
    spectral_radius = float(max(abs(np.linalg.eigvals(q))))
    if spectral_radius >= 1.0 - 1e-12:
        raise ValueError(
            "Transient submatrix is not absorbing/stable; fundamental matrix is undefined"
        )
    fundamental = np.linalg.inv(np.eye(len(TRANSIENT_STATES)) - q)
    start_row = TRANSIENT_STATES.index(start_state)
    values = fundamental[start_row]
    return {
        state: float(values[idx])
        for idx, state in enumerate(TRANSIENT_STATES)
    }


def expected_resource_days_by_cohort(instance: HealthcareInstance) -> pd.DataFrame:
    """Return expected ICU/Ward bed-days per arrival for each cohort."""
    matrices = load_transition_matrices(instance.transitions)
    rows: list[dict[str, object]] = []
    for cohort, matrix in sorted(matrices.items()):
        visits = expected_transient_visits(matrix, start_state="ED")
        for resource in RESOURCE_STATES:
            rows.append(
                {
                    "cohort": str(cohort),
                    "resource": resource,
                    "expected_resource_days_per_arrival": float(visits[resource]),
                    "unit": "bed_days_per_arrival",
                    "method": "absorbing_markov_fundamental_matrix",
                }
            )
    return pd.DataFrame(rows).sort_values(["cohort", "resource"]).reset_index(drop=True)


def _validate_forecast_frame(forecast_df: pd.DataFrame, quantile_column: str) -> None:
    required = {"hospital_id", "cohort", quantile_column}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"Forecast frame missing columns: {sorted(missing)}")
    values = pd.to_numeric(forecast_df[quantile_column], errors="raise")
    if (values < 0).any():
        raise ValueError("Forecast arrivals must be nonnegative")


def build_resource_pressure_projection(
    instance: HealthcareInstance,
    forecast_df: pd.DataFrame,
    *,
    quantile_column: str,
) -> pd.DataFrame:
    """Translate arrival forecasts into ICU/Ward census and capacity pressure.

    The returned `expected_census` is a Little's-Law steady-state approximation
    using forecast patients/day times expected bed-days/patient.
    """
    _validate_forecast_frame(forecast_df, quantile_column)
    resource_days = expected_resource_days_by_cohort(instance)

    work = forecast_df[["hospital_id", "cohort", quantile_column]].copy()
    work[quantile_column] = pd.to_numeric(work[quantile_column], errors="raise").astype(float)
    merged = work.merge(resource_days, on="cohort", how="left", validate="many_to_many")
    if merged["expected_resource_days_per_arrival"].isna().any():
        raise ValueError("Forecast contains a cohort without a transition matrix")
    merged["expected_bed_days"] = (
        merged[quantile_column] * merged["expected_resource_days_per_arrival"]
    )

    census = (
        merged.groupby(["hospital_id", "resource"], as_index=False)["expected_bed_days"]
        .sum()
        .rename(columns={"expected_bed_days": "expected_census"})
    )

    capacity = instance.capacities.df[["hospital_id", "resource", "base_capacity"]].copy()
    thresholds = instance.safe_thresholds.df[
        ["hospital_id", "resource", "safe_utilization"]
    ].copy()
    result = census.merge(
        capacity,
        on=["hospital_id", "resource"],
        how="left",
        validate="one_to_one",
    ).merge(
        thresholds,
        on=["hospital_id", "resource"],
        how="left",
        validate="one_to_one",
    )
    if result[["base_capacity", "safe_utilization"]].isna().any().any():
        raise ValueError("Capacity or safe-utilization data missing for projected resource")

    result["safe_capacity"] = result["base_capacity"] * result["safe_utilization"]
    result["projected_utilization"] = result["expected_census"] / result["base_capacity"]
    result["safe_capacity_gap"] = result["expected_census"] - result["safe_capacity"]
    result["above_safe_capacity"] = result["safe_capacity_gap"] > 0.0
    result["source_quantile"] = quantile_column
    result["calculation_label"] = "CALCULATED"
    result["units"] = "patients_or_beds"
    return result.sort_values(["hospital_id", "resource"]).reset_index(drop=True)
