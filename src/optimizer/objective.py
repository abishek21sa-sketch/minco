from __future__ import annotations

from typing import Dict, List, Tuple

import gurobipy as gp


Arc = Tuple[str, str]


def _build_cost_lookup(costs_df) -> Dict[str, float]:
    """
    Convert costs table into a simple dictionary:
        cost_name -> value
    """
    return dict(zip(costs_df["cost_name"], costs_df["value"]))


def set_minimal_capacity_objective(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
    validated_forecast_df,
    costs_df,
) -> gp.LinExpr:
    """
    Set the minimal capacity-response objective.

    Objective terms
    ---------------
    - surge activation cost
    - unsafe utilization penalty
    - overflow penalty
    """
    u = variables["u"]
    z = variables["z"]
    o = variables["o"]

    cost_lookup = _build_cost_lookup(costs_df)

    surge_cost = float(cost_lookup["surge_bed"])
    unsafe_cost_by_resource = {
        "ICU": float(cost_lookup["unsafe_util_icu"]),
        "Ward": float(cost_lookup["unsafe_util_ward"]),
    }
    overflow_cost_by_resource = {
        "ICU": float(cost_lookup["overflow_icu"]),
        "Ward": float(cost_lookup["overflow_ward"]),
    }

    unique_keys = (
        validated_forecast_df[["day", "hospital_id", "resource"]]
        .drop_duplicates()
        .sort_values(["day", "hospital_id", "resource"])
        .itertuples(index=False)
    )

    objective = gp.quicksum(
        surge_cost * u[row.hospital_id, row.resource, row.day]
        + unsafe_cost_by_resource[row.resource] * z[row.hospital_id, row.resource, row.day]
        + overflow_cost_by_resource[row.resource] * o[row.hospital_id, row.resource, row.day]
        for row in unique_keys
    )

    model.setObjective(objective, gp.GRB.MINIMIZE)
    return objective


def set_extended_capacity_objective(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
    validated_forecast_df,
    elective_bounds_df,
    costs_df,
) -> gp.LinExpr:
    """
    Set the extended capacity-response objective with elective rejection.

    Objective terms
    ---------------
    - surge activation cost
    - unsafe utilization penalty
    - overflow penalty
    - elective rejection penalty
    """
    u = variables["u"]
    z = variables["z"]
    o = variables["o"]
    r = variables["r"]

    cost_lookup = _build_cost_lookup(costs_df)

    surge_cost = float(cost_lookup["surge_bed"])
    elective_rejection_cost = float(cost_lookup["elective_rejection"])

    unsafe_cost_by_resource = {
        "ICU": float(cost_lookup["unsafe_util_icu"]),
        "Ward": float(cost_lookup["unsafe_util_ward"]),
    }
    overflow_cost_by_resource = {
        "ICU": float(cost_lookup["overflow_icu"]),
        "Ward": float(cost_lookup["overflow_ward"]),
    }

    capacity_keys = (
        validated_forecast_df[["day", "hospital_id", "resource"]]
        .drop_duplicates()
        .sort_values(["day", "hospital_id", "resource"])
        .itertuples(index=False)
    )

    elective_keys = (
        elective_bounds_df[elective_bounds_df["cohort"] == "c3"][
            ["day", "hospital_id"]
        ]
        .drop_duplicates()
        .sort_values(["day", "hospital_id"])
        .itertuples(index=False)
    )

    capacity_term = gp.quicksum(
        surge_cost * u[row.hospital_id, row.resource, row.day]
        + unsafe_cost_by_resource[row.resource] * z[row.hospital_id, row.resource, row.day]
        + overflow_cost_by_resource[row.resource] * o[row.hospital_id, row.resource, row.day]
        for row in capacity_keys
    )

    elective_term = gp.quicksum(
        elective_rejection_cost * r[row.hospital_id, row.day]
        for row in elective_keys
    )

    objective = capacity_term + elective_term
    model.setObjective(objective, gp.GRB.MINIMIZE)
    return objective


def set_network_capacity_objective(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
    validated_forecast_df,
    elective_bounds_df,
    transfer_cost_lookup: Dict[Arc, float],
    days: List[int],
    costs_df,
) -> gp.LinExpr:
    """
    Set the network capacity-response objective with:
    - surge activation cost
    - unsafe utilization penalty
    - overflow penalty
    - elective rejection penalty
    - ICU transfer cost

    Parameters
    ----------
    transfer_cost_lookup:
        Mapping (from_hospital, to_hospital) -> effective transfer cost
    days:
        Planning days used to index transfer variables x[i,j,t]
    """
    u = variables["u"]
    z = variables["z"]
    o = variables["o"]
    r = variables["r"]
    x = variables["x"]

    cost_lookup = _build_cost_lookup(costs_df)

    surge_cost = float(cost_lookup["surge_bed"])
    elective_rejection_cost = float(cost_lookup["elective_rejection"])

    unsafe_cost_by_resource = {
        "ICU": float(cost_lookup["unsafe_util_icu"]),
        "Ward": float(cost_lookup["unsafe_util_ward"]),
    }
    overflow_cost_by_resource = {
        "ICU": float(cost_lookup["overflow_icu"]),
        "Ward": float(cost_lookup["overflow_ward"]),
    }

    capacity_keys = (
        validated_forecast_df[["day", "hospital_id", "resource"]]
        .drop_duplicates()
        .sort_values(["day", "hospital_id", "resource"])
        .itertuples(index=False)
    )

    elective_keys = (
        elective_bounds_df[elective_bounds_df["cohort"] == "c3"][
            ["day", "hospital_id"]
        ]
        .drop_duplicates()
        .sort_values(["day", "hospital_id"])
        .itertuples(index=False)
    )

    capacity_term = gp.quicksum(
        surge_cost * u[row.hospital_id, row.resource, row.day]
        + unsafe_cost_by_resource[row.resource] * z[row.hospital_id, row.resource, row.day]
        + overflow_cost_by_resource[row.resource] * o[row.hospital_id, row.resource, row.day]
        for row in capacity_keys
    )

    elective_term = gp.quicksum(
        elective_rejection_cost * r[row.hospital_id, row.day]
        for row in elective_keys
    )

    transfer_term = gp.quicksum(
        transfer_cost_lookup[(i, j)] * x[i, j, t]
        for (i, j) in transfer_cost_lookup.keys()
        for t in days
    )

    objective = capacity_term + elective_term + transfer_term
    model.setObjective(objective, gp.GRB.MINIMIZE)
    return objective