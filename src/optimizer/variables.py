from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import gurobipy as gp
from gurobipy import GRB


Arc = Tuple[str, str]


def add_capacity_response_variables(
    model: gp.Model,
    days: Iterable[int],
    hospitals: Iterable[str],
    resources: Iterable[str],
) -> Dict[str, gp.tupledict]:
    """
    Add the minimal capacity-response decision variables to the model.

    Variables
    ---------
    u[h, r, t] : continuous
        Surge capacity activated at hospital h for resource r on day t.

    z[h, r, t] : continuous
        Unsafe-load slack above safe capacity at hospital h for resource r on day t.

    o[h, r, t] : continuous
        Overflow slack above base capacity at hospital h for resource r on day t.
    """
    u = model.addVars(
        hospitals,
        resources,
        days,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="u",
    )

    z = model.addVars(
        hospitals,
        resources,
        days,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="z",
    )

    o = model.addVars(
        hospitals,
        resources,
        days,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="o",
    )

    return {
        "u": u,
        "z": z,
        "o": o,
    }


def add_elective_decision_variables(
    model: gp.Model,
    days: Iterable[int],
    hospitals: Iterable[str],
) -> Dict[str, gp.tupledict]:
    """
    Add elective-control decision variables for cohort c3.

    Variables
    ---------
    e[h, t] : continuous
        Accepted elective c3 admissions at hospital h on day t.

    r[h, t] : continuous
        Rejected / deferred elective c3 admissions at hospital h on day t.
    """
    e = model.addVars(
        hospitals,
        days,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="e",
    )

    r = model.addVars(
        hospitals,
        days,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="r",
    )

    return {
        "e": e,
        "r": r,
    }


def add_transfer_variables(
    model: gp.Model,
    days: Iterable[int],
    arcs: List[Arc],
) -> Dict[str, gp.tupledict]:
    """
    Add ICU transfer decision variables.

    Variables
    ---------
    x[i, j, t] : continuous
        Expected ICU load transferred from hospital i to hospital j on day t.
    """
    x = model.addVars(
        arcs,
        days,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x",
    )

    return {
        "x": x,
    }


def add_extended_capacity_response_variables(
    model: gp.Model,
    days: Iterable[int],
    hospitals: Iterable[str],
    resources: Iterable[str],
) -> Dict[str, gp.tupledict]:
    """
    Add all decision variables for the elective-control capacity model.

    Variable families
    -----------------
    u[h, r, t] : surge activated
    z[h, r, t] : unsafe slack
    o[h, r, t] : overflow slack
    e[h, t]    : accepted elective c3 admissions
    r[h, t]    : rejected elective c3 admissions
    """
    capacity_vars = add_capacity_response_variables(
        model=model,
        days=days,
        hospitals=hospitals,
        resources=resources,
    )

    elective_vars = add_elective_decision_variables(
        model=model,
        days=days,
        hospitals=hospitals,
    )

    return {
        **capacity_vars,
        **elective_vars,
    }


def add_network_capacity_response_variables(
    model: gp.Model,
    days: Iterable[int],
    hospitals: Iterable[str],
    resources: Iterable[str],
    arcs: List[Arc],
) -> Dict[str, gp.tupledict]:
    """
    Add all decision variables for the networked capacity model.

    Variable families
    -----------------
    u[h, r, t] : surge activated
    z[h, r, t] : unsafe slack
    o[h, r, t] : overflow slack
    e[h, t]    : accepted elective c3 admissions
    r[h, t]    : rejected elective c3 admissions
    x[i, j, t] : ICU load transferred along allowed arc (i, j) on day t
    """
    extended_vars = add_extended_capacity_response_variables(
        model=model,
        days=days,
        hospitals=hospitals,
        resources=resources,
    )

    transfer_vars = add_transfer_variables(
        model=model,
        days=days,
        arcs=arcs,
    )

    return {
        **extended_vars,
        **transfer_vars,
    }


def extract_variable_keys(
    validated_forecast_df,
) -> Tuple[list[int], list[str], list[str]]:
    """
    Extract canonical sets for the optimizer from the validated forecast table.

    Expected columns in validated_forecast_df:
        day, hospital_id, resource

    Returns
    -------
    tuple
        (days, hospitals, resources)
    """
    days = sorted(validated_forecast_df["day"].unique().tolist())
    hospitals = sorted(validated_forecast_df["hospital_id"].unique().tolist())
    resources = sorted(validated_forecast_df["resource"].unique().tolist())

    return days, hospitals, resources