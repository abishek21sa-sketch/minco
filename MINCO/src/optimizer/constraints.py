from __future__ import annotations

from typing import Dict, List, Tuple

import gurobipy as gp
import pandas as pd


Arc = Tuple[str, str]


def add_minimal_capacity_constraints(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
    validated_forecast_df,
) -> None:
    """
    Add the minimal capacity-response constraints.

    Variables
    ---------
    u[h, r, t] : surge activated
    z[h, r, t] : unsafe slack above safe capacity
    o[h, r, t] : overflow slack above base capacity
    """
    u = variables["u"]
    z = variables["z"]
    o = variables["o"]

    rows = (
        validated_forecast_df[
            [
                "day",
                "hospital_id",
                "resource",
                "expected_demand",
                "safe_capacity",
                "base_capacity",
                "max_surge",
            ]
        ]
        .drop_duplicates()
        .sort_values(["day", "hospital_id", "resource"])
        .itertuples(index=False)
    )

    for row in rows:
        h = row.hospital_id
        r = row.resource
        t = int(row.day)

        demand = float(row.expected_demand)
        safe_capacity = float(row.safe_capacity)
        base_capacity = float(row.base_capacity)
        max_surge = float(row.max_surge)

        model.addConstr(
            u[h, r, t] <= max_surge,
            name=f"surge_cap[{h},{r},{t}]",
        )

        model.addConstr(
            demand <= safe_capacity + u[h, r, t] + z[h, r, t] + o[h, r, t],
            name=f"safe_cover[{h},{r},{t}]",
        )

        model.addConstr(
            demand <= base_capacity + u[h, r, t] + o[h, r, t],
            name=f"base_cover[{h},{r},{t}]",
        )


def _build_elective_bounds_lookup(elective_bounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only c3 elective rows and ensure one row per hospital-day.
    """
    filtered = elective_bounds_df[elective_bounds_df["cohort"] == "c3"].copy()

    filtered = (
        filtered[
            ["day", "hospital_id", "min_elective", "max_elective"]
        ]
        .drop_duplicates()
        .sort_values(["day", "hospital_id"])
        .reset_index(drop=True)
    )

    return filtered


def add_extended_capacity_constraints(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
    validated_forecast_df: pd.DataFrame,
    elective_bounds_df: pd.DataFrame,
    elective_response_df: pd.DataFrame,
) -> None:
    """
    Add the elective-control capacity constraints.

    Variables
    ---------
    u[h, r, t] : surge activated
    z[h, r, t] : unsafe slack above safe capacity
    o[h, r, t] : overflow slack above base capacity
    e[h, t]    : accepted elective c3 admissions
    r[h, t]    : rejected elective c3 admissions
    """
    u = variables["u"]
    z = variables["z"]
    o = variables["o"]
    e = variables["e"]
    reject = variables["r"]

    elective_bounds = _build_elective_bounds_lookup(elective_bounds_df)

    capacity_rows = (
        validated_forecast_df[
            [
                "day",
                "hospital_id",
                "resource",
                "expected_demand",
                "safe_capacity",
                "base_capacity",
                "max_surge",
            ]
        ]
        .drop_duplicates()
        .sort_values(["day", "hospital_id", "resource"])
        .reset_index(drop=True)
    )

    for row in elective_bounds.itertuples(index=False):
        h = row.hospital_id
        t = int(row.day)
        min_elective = float(row.min_elective)
        max_elective = float(row.max_elective)

        model.addConstr(
            e[h, t] + reject[h, t] == max_elective,
            name=f"elective_balance[{h},{t}]",
        )

        model.addConstr(
            e[h, t] >= min_elective,
            name=f"elective_min[{h},{t}]",
        )

    response_lookup = (
        elective_response_df.groupby(
            ["hospital_id", "admit_day", "day", "resource"], as_index=False
        )["marginal_demand"]
        .sum()
    )

    grouped_response = {}
    for row in response_lookup.itertuples(index=False):
        key = (str(row.hospital_id), int(row.day), str(row.resource))
        grouped_response.setdefault(key, []).append(
            (int(row.admit_day), float(row.marginal_demand))
        )

    for row in capacity_rows.itertuples(index=False):
        h = row.hospital_id
        r = row.resource
        t = int(row.day)

        fixed_demand = float(row.expected_demand)
        safe_capacity = float(row.safe_capacity)
        base_capacity = float(row.base_capacity)
        max_surge = float(row.max_surge)

        model.addConstr(
            u[h, r, t] <= max_surge,
            name=f"surge_cap[{h},{r},{t}]",
        )

        response_terms = grouped_response.get((h, t, r), [])

        elective_induced_demand = gp.quicksum(
            coeff * e[h, admit_day]
            for admit_day, coeff in response_terms
        )

        total_demand_expr = fixed_demand + elective_induced_demand

        model.addConstr(
            total_demand_expr <= safe_capacity + u[h, r, t] + z[h, r, t] + o[h, r, t],
            name=f"safe_cover_ext[{h},{r},{t}]",
        )

        model.addConstr(
            total_demand_expr <= base_capacity + u[h, r, t] + o[h, r, t],
            name=f"base_cover_ext[{h},{r},{t}]",
        )


def add_network_capacity_constraints(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
    validated_forecast_df: pd.DataFrame,
    elective_bounds_df: pd.DataFrame,
    elective_response_df: pd.DataFrame,
    arcs: List[Arc],
    transfer_capacity_lookup: Dict[Arc, float],
) -> None:
    """
    Add the networked capacity constraints with ICU transfers.

    Variables
    ---------
    u[h, r, t] : surge activated
    z[h, r, t] : unsafe slack
    o[h, r, t] : overflow slack
    e[h, t]    : accepted elective c3 admissions
    r[h, t]    : rejected elective c3 admissions
    x[i, j, t] : ICU load transferred from i to j on day t

    Logic
    -----
    Ward:
        demand = fixed + elective_induced

    ICU:
        demand = fixed + elective_induced - outbound_transfer + inbound_transfer

    Also enforce:
        x[i,j,t] <= transfer_capacity[i,j]
        outbound_transfer <= local ICU pre-transfer demand

    ``allowed`` controls topology; ``transfer_capacity`` controls the amount
    that can move across an enabled lane during one planning day.
    """
    u = variables["u"]
    z = variables["z"]
    o = variables["o"]
    e = variables["e"]
    reject = variables["r"]
    x = variables["x"]

    elective_bounds = _build_elective_bounds_lookup(elective_bounds_df)

    capacity_rows = (
        validated_forecast_df[
            [
                "day",
                "hospital_id",
                "resource",
                "expected_demand",
                "safe_capacity",
                "base_capacity",
                "max_surge",
            ]
        ]
        .drop_duplicates()
        .sort_values(["day", "hospital_id", "resource"])
        .reset_index(drop=True)
    )

    for row in elective_bounds.itertuples(index=False):
        h = row.hospital_id
        t = int(row.day)
        min_elective = float(row.min_elective)
        max_elective = float(row.max_elective)

        model.addConstr(
            e[h, t] + reject[h, t] == max_elective,
            name=f"elective_balance[{h},{t}]",
        )

        model.addConstr(
            e[h, t] >= min_elective,
            name=f"elective_min[{h},{t}]",
        )

    response_lookup = (
        elective_response_df.groupby(
            ["hospital_id", "admit_day", "day", "resource"], as_index=False
        )["marginal_demand"]
        .sum()
    )

    grouped_response = {}
    for row in response_lookup.itertuples(index=False):
        key = (str(row.hospital_id), int(row.day), str(row.resource))
        grouped_response.setdefault(key, []).append(
            (int(row.admit_day), float(row.marginal_demand))
        )

    outbound_map = {}
    inbound_map = {}
    for i, j in arcs:
        outbound_map.setdefault(i, []).append((i, j))
        inbound_map.setdefault(j, []).append((i, j))

    days = sorted(int(day) for day in capacity_rows["day"].unique())
    for i, j in arcs:
        if (i, j) not in transfer_capacity_lookup:
            raise ValueError(f"Missing transfer capacity for allowed arc {(i, j)}")
        lane_capacity = float(transfer_capacity_lookup[(i, j)])
        for t in days:
            model.addConstr(
                x[i, j, t] <= lane_capacity,
                name=f"transfer_capacity[{i},{j},{t}]",
            )

    for row in capacity_rows.itertuples(index=False):
        h = row.hospital_id
        r = row.resource
        t = int(row.day)

        fixed_demand = float(row.expected_demand)
        safe_capacity = float(row.safe_capacity)
        base_capacity = float(row.base_capacity)
        max_surge = float(row.max_surge)

        model.addConstr(
            u[h, r, t] <= max_surge,
            name=f"surge_cap_net[{h},{r},{t}]",
        )

        response_terms = grouped_response.get((h, t, r), [])

        elective_induced_demand = gp.quicksum(
            coeff * e[h, admit_day]
            for admit_day, coeff in response_terms
        )

        local_pretransfer_demand = fixed_demand + elective_induced_demand

        if r == "ICU":
            outbound = gp.quicksum(
                x[i, j, t]
                for (i, j) in outbound_map.get(h, [])
            )

            inbound = gp.quicksum(
                x[i, j, t]
                for (i, j) in inbound_map.get(h, [])
            )

            model.addConstr(
                outbound <= local_pretransfer_demand,
                name=f"icu_transfer_limit[{h},{t}]",
            )

            adjusted_demand = local_pretransfer_demand - outbound + inbound
        else:
            adjusted_demand = local_pretransfer_demand

        model.addConstr(
            adjusted_demand <= safe_capacity + u[h, r, t] + z[h, r, t] + o[h, r, t],
            name=f"safe_cover_net[{h},{r},{t}]",
        )

        model.addConstr(
            adjusted_demand <= base_capacity + u[h, r, t] + o[h, r, t],
            name=f"base_cover_net[{h},{r},{t}]",
        )