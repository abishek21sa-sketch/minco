"""Reusable feasibility and regression checks for solved optimization snapshots."""
from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from src.config.dataclasses import HealthcareInstance


def summarize_solved_model(model: Any) -> dict[str, float]:
    """Extract stable diagnostics from a solved Gurobi model."""
    if int(model.SolCount) < 1:
        raise AssertionError("Optimization model has no feasible solution")
    objective = float(model.ObjVal)
    if not math.isfinite(objective):
        raise AssertionError("Optimization objective is not finite")
    return {
        "status": float(model.Status),
        "solution_count": float(model.SolCount),
        "objective_value": objective,
        "runtime_seconds": float(model.Runtime),
        "variables": float(model.NumVars),
        "constraints": float(model.NumConstrs),
        "constraint_violation": float(getattr(model, "ConstrVio", 0.0)),
        "bound_violation": float(getattr(model, "BoundVio", 0.0)),
    }


def assert_optimal_model(model: Any, *, feasibility_tolerance: float = 1e-6) -> dict[str, float]:
    diagnostics = summarize_solved_model(model)
    if int(diagnostics["status"]) != 2:
        raise AssertionError(f"Expected Gurobi OPTIMAL status=2, got {diagnostics['status']}")
    if diagnostics["constraint_violation"] > feasibility_tolerance:
        raise AssertionError(
            f"Constraint violation {diagnostics['constraint_violation']} exceeds tolerance"
        )
    if diagnostics["bound_violation"] > feasibility_tolerance:
        raise AssertionError(f"Bound violation {diagnostics['bound_violation']} exceeds tolerance")
    return diagnostics


def assert_objective_repeatable(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    absolute_tolerance: float = 1e-6,
) -> None:
    difference = abs(float(first["objective_value"]) - float(second["objective_value"]))
    if difference > absolute_tolerance:
        raise AssertionError(
            f"Objective changed by {difference}, exceeding tolerance {absolute_tolerance}"
        )


def _assert_nonnegative(frame: pd.DataFrame, column: str, tolerance: float) -> None:
    if column not in frame.columns:
        raise AssertionError(f"Missing solution column: {column}")
    minimum = float(frame[column].min()) if not frame.empty else 0.0
    if minimum < -tolerance:
        raise AssertionError(f"{column} contains negative value {minimum}")


def validate_network_solution_tables(
    snapshot: Mapping[str, pd.DataFrame],
    instance: HealthcareInstance,
    *,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Verify non-negativity, surge bounds, elective balance, and allowed transfer arcs."""
    required = {
        "surge": "surge_activated",
        "unsafe_slack": "unsafe_slack",
        "overflow_slack": "overflow_slack",
        "elective_accepted": "elective_accepted",
        "elective_rejected": "elective_rejected",
        "icu_transfers": "icu_transfer_load",
    }
    for table_name, value_col in required.items():
        if table_name not in snapshot:
            raise AssertionError(f"Missing solution table: {table_name}")
        _assert_nonnegative(snapshot[table_name], value_col, tolerance)

    surge = snapshot["surge"].merge(
        instance.surge_caps.df,
        on=["hospital_id", "resource"],
        how="left",
        validate="many_to_one",
    )
    if surge["max_surge"].isna().any():
        raise AssertionError("Surge solution contains unknown hospital-resource rows")
    surge_excess = float((surge["surge_activated"] - surge["max_surge"]).max())
    if surge_excess > tolerance:
        raise AssertionError(f"Surge capacity exceeded by {surge_excess}")

    elective = snapshot["elective_accepted"].merge(
        snapshot["elective_rejected"],
        on=["hospital_id", "day"],
        how="outer",
        validate="one_to_one",
    ).merge(
        instance.elective_bounds.df.query("cohort == 'c3'")[
            ["hospital_id", "day", "min_elective", "max_elective"]
        ],
        on=["hospital_id", "day"],
        how="left",
        validate="one_to_one",
    )
    if elective[["min_elective", "max_elective"]].isna().any().any():
        raise AssertionError("Elective solution contains unknown hospital-day rows")
    balance_error = float(
        (
            elective["elective_accepted"]
            + elective["elective_rejected"]
            - elective["max_elective"]
        ).abs().max()
    )
    minimum_error = float((elective["min_elective"] - elective["elective_accepted"]).max())
    if balance_error > tolerance:
        raise AssertionError(f"Elective balance error {balance_error} exceeds tolerance")
    if minimum_error > tolerance:
        raise AssertionError(f"Elective minimum shortfall {minimum_error} exceeds tolerance")

    allowed_arcs = set(
        instance.transfer_lanes.df.loc[
            instance.transfer_lanes.df["allowed"] == 1,
            ["from_hospital", "to_hospital"],
        ].itertuples(index=False, name=None)
    )
    transfer_arcs = set(
        snapshot["icu_transfers"][["from_hospital", "to_hospital"]].itertuples(
            index=False, name=None
        )
    )
    disallowed = sorted(transfer_arcs - allowed_arcs)
    if disallowed:
        raise AssertionError(f"Solution contains disallowed transfer arcs: {disallowed}")

    lane_caps = instance.transfer_lanes.df.loc[
        instance.transfer_lanes.df["allowed"] == 1,
        ["from_hospital", "to_hospital", "transfer_capacity"],
    ].copy()
    transfer_with_caps = snapshot["icu_transfers"].merge(
        lane_caps,
        on=["from_hospital", "to_hospital"],
        how="left",
        validate="many_to_one",
    )
    if transfer_with_caps["transfer_capacity"].isna().any():
        raise AssertionError("Transfer solution contains an arc with no capacity definition")
    transfer_capacity_excess = float(
        (
            transfer_with_caps["icu_transfer_load"]
            - transfer_with_caps["transfer_capacity"]
        ).max()
    ) if not transfer_with_caps.empty else 0.0
    if transfer_capacity_excess > tolerance:
        raise AssertionError(
            f"Transfer lane capacity exceeded by {transfer_capacity_excess}"
        )

    return {
        "surge_bound_max_excess": max(0.0, surge_excess),
        "elective_balance_max_error": balance_error,
        "elective_minimum_max_shortfall": max(0.0, minimum_error),
        "transfer_arc_count": float(len(transfer_arcs)),
        "transfer_capacity_max_excess": max(0.0, transfer_capacity_excess),
    }
