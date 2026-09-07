from __future__ import annotations

from typing import Dict, List, Tuple

import gurobipy as gp
import pandas as pd


Arc = Tuple[str, str]


def _extract_capacity_tupledict_solution(
    var_dict: gp.tupledict,
    value_name: str,
) -> pd.DataFrame:
    """
    Convert a Gurobi tupledict solution into a tidy DataFrame.

    Expected variable keys:
        (hospital_id, resource, day)
    """
    records = []

    for key, var in var_dict.items():
        hospital_id, resource, day = key
        value = float(var.X) if var.X is not None else 0.0

        records.append(
            {
                "hospital_id": str(hospital_id),
                "resource": str(resource),
                "day": int(day),
                value_name: value,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["day", "hospital_id", "resource"]).reset_index(drop=True)

    return df


def _extract_elective_tupledict_solution(
    var_dict: gp.tupledict,
    value_name: str,
) -> pd.DataFrame:
    """
    Convert a Gurobi tupledict with keys (hospital_id, day)
    into a tidy DataFrame.
    """
    records = []

    for key, var in var_dict.items():
        hospital_id, day = key
        value = float(var.X) if var.X is not None else 0.0

        records.append(
            {
                "hospital_id": str(hospital_id),
                "day": int(day),
                value_name: value,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["day", "hospital_id"]).reset_index(drop=True)

    return df


def _extract_transfer_tupledict_solution(
    var_dict: gp.tupledict,
    value_name: str,
) -> pd.DataFrame:
    """
    Convert a Gurobi tupledict with keys (from_hospital, to_hospital, day)
    into a tidy DataFrame.
    """
    records = []

    for key, var in var_dict.items():
        from_hospital, to_hospital, day = key
        value = float(var.X) if var.X is not None else 0.0

        records.append(
            {
                "from_hospital": str(from_hospital),
                "to_hospital": str(to_hospital),
                "day": int(day),
                value_name: value,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["day", "from_hospital", "to_hospital"]).reset_index(drop=True)

    return df


def extract_minimal_capacity_solution(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
) -> Dict[str, pd.DataFrame]:
    """
    Extract the solved values of the minimal capacity-response variables.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys:
            surge
            unsafe_slack
            overflow_slack
    """
    if model.SolCount == 0:
        raise ValueError("Model has no solution available to extract.")

    surge_df = _extract_capacity_tupledict_solution(
        variables["u"],
        value_name="surge_activated",
    )

    unsafe_df = _extract_capacity_tupledict_solution(
        variables["z"],
        value_name="unsafe_slack",
    )

    overflow_df = _extract_capacity_tupledict_solution(
        variables["o"],
        value_name="overflow_slack",
    )

    return {
        "surge": surge_df,
        "unsafe_slack": unsafe_df,
        "overflow_slack": overflow_df,
    }


def extract_extended_capacity_solution(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
) -> Dict[str, pd.DataFrame]:
    """
    Extract the solved values of the extended capacity-response variables.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys:
            surge
            unsafe_slack
            overflow_slack
            elective_accepted
            elective_rejected
    """
    if model.SolCount == 0:
        raise ValueError("Model has no solution available to extract.")

    capacity_tables = extract_minimal_capacity_solution(model, variables)

    accepted_df = _extract_elective_tupledict_solution(
        variables["e"],
        value_name="elective_accepted",
    )

    rejected_df = _extract_elective_tupledict_solution(
        variables["r"],
        value_name="elective_rejected",
    )

    return {
        **capacity_tables,
        "elective_accepted": accepted_df,
        "elective_rejected": rejected_df,
    }


def extract_network_capacity_solution(
    model: gp.Model,
    variables: Dict[str, gp.tupledict],
    transfer_cost_lookup: Dict[Arc, float] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Extract the solved values of the network capacity-response variables.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys:
            surge
            unsafe_slack
            overflow_slack
            elective_accepted
            elective_rejected
            icu_transfers
    """
    if model.SolCount == 0:
        raise ValueError("Model has no solution available to extract.")

    extended_tables = extract_extended_capacity_solution(model, variables)

    transfer_df = _extract_transfer_tupledict_solution(
        variables["x"],
        value_name="icu_transfer_load",
    )

    if transfer_cost_lookup is not None and not transfer_df.empty:
        transfer_df["transfer_cost"] = transfer_df.apply(
            lambda row: float(
                transfer_cost_lookup[(row["from_hospital"], row["to_hospital"])]
            ),
            axis=1,
        )
        transfer_df["transfer_cost_incurred"] = (
            transfer_df["icu_transfer_load"] * transfer_df["transfer_cost"]
        )

    return {
        **extended_tables,
        "icu_transfers": transfer_df,
    }


def build_minimal_capacity_summary(
    validated_forecast_df: pd.DataFrame,
    solution_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge forecast validation output with optimization decisions
    for the minimal model.
    """
    summary = validated_forecast_df.copy()

    summary = summary.merge(
        solution_tables["surge"],
        on=["day", "hospital_id", "resource"],
        how="left",
        validate="one_to_one",
    )

    summary = summary.merge(
        solution_tables["unsafe_slack"],
        on=["day", "hospital_id", "resource"],
        how="left",
        validate="one_to_one",
    )

    summary = summary.merge(
        solution_tables["overflow_slack"],
        on=["day", "hospital_id", "resource"],
        how="left",
        validate="one_to_one",
    )

    for col in ["surge_activated", "unsafe_slack", "overflow_slack"]:
        summary[col] = summary[col].fillna(0.0)

    summary["effective_safe_plus_surge"] = (
        summary["safe_capacity"] + summary["surge_activated"]
    )
    summary["effective_base_plus_surge"] = (
        summary["base_capacity"] + summary["surge_activated"]
    )

    summary = summary.sort_values(
        ["day", "hospital_id", "resource"]
    ).reset_index(drop=True)

    return summary


def build_extended_capacity_summary(
    validated_forecast_df: pd.DataFrame,
    solution_tables: Dict[str, pd.DataFrame],
    elective_bounds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge forecast validation output with optimization decisions
    for the extended elective-control model.

    Adds accepted and rejected electives at the hospital-day level.
    These are repeated across resources after merging, which is fine
    for reporting.
    """
    summary = build_minimal_capacity_summary(
        validated_forecast_df=validated_forecast_df,
        solution_tables=solution_tables,
    )

    elective_bounds = (
        elective_bounds_df[elective_bounds_df["cohort"] == "c3"][
            ["day", "hospital_id", "min_elective", "max_elective"]
        ]
        .drop_duplicates()
        .sort_values(["day", "hospital_id"])
        .reset_index(drop=True)
    )

    summary = summary.merge(
        elective_bounds,
        on=["day", "hospital_id"],
        how="left",
        validate="many_to_one",
    )

    summary = summary.merge(
        solution_tables["elective_accepted"],
        on=["day", "hospital_id"],
        how="left",
        validate="many_to_one",
    )

    summary = summary.merge(
        solution_tables["elective_rejected"],
        on=["day", "hospital_id"],
        how="left",
        validate="many_to_one",
    )

    for col in ["elective_accepted", "elective_rejected"]:
        summary[col] = summary[col].fillna(0.0)

    summary = summary.sort_values(
        ["day", "hospital_id", "resource"]
    ).reset_index(drop=True)

    return summary


def build_network_capacity_summary(
    validated_forecast_df: pd.DataFrame,
    solution_tables: Dict[str, pd.DataFrame],
    elective_bounds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the network-capacity summary.

    Returns a hospital-day-resource table with:
    - forecast/capacity info
    - surge / unsafe / overflow decisions
    - elective decisions
    - inbound / outbound / net ICU transfers
    """
    summary = build_extended_capacity_summary(
        validated_forecast_df=validated_forecast_df,
        solution_tables=solution_tables,
        elective_bounds_df=elective_bounds_df,
    )

    transfer_df = solution_tables["icu_transfers"].copy()

    if transfer_df.empty:
        summary["icu_outbound_transfer"] = 0.0
        summary["icu_inbound_transfer"] = 0.0
        summary["icu_net_transfer"] = 0.0
        return summary

    outbound = (
        transfer_df.groupby(["day", "from_hospital"], as_index=False)["icu_transfer_load"]
        .sum()
        .rename(
            columns={
                "from_hospital": "hospital_id",
                "icu_transfer_load": "icu_outbound_transfer",
            }
        )
    )

    inbound = (
        transfer_df.groupby(["day", "to_hospital"], as_index=False)["icu_transfer_load"]
        .sum()
        .rename(
            columns={
                "to_hospital": "hospital_id",
                "icu_transfer_load": "icu_inbound_transfer",
            }
        )
    )

    summary = summary.merge(
        outbound,
        on=["day", "hospital_id"],
        how="left",
        validate="many_to_one",
    )

    summary = summary.merge(
        inbound,
        on=["day", "hospital_id"],
        how="left",
        validate="many_to_one",
    )

    summary["icu_outbound_transfer"] = summary["icu_outbound_transfer"].fillna(0.0)
    summary["icu_inbound_transfer"] = summary["icu_inbound_transfer"].fillna(0.0)
    summary["icu_net_transfer"] = (
        summary["icu_inbound_transfer"] - summary["icu_outbound_transfer"]
    )

    summary = summary.sort_values(
        ["day", "hospital_id", "resource"]
    ).reset_index(drop=True)

    return summary


def summarize_nonzero_actions(
    summary_df: pd.DataFrame,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Filter to rows where the optimizer took a nonzero action
    or where nonzero slack remains.
    """
    candidate_cols = [
        "surge_activated",
        "unsafe_slack",
        "overflow_slack",
        "elective_accepted",
        "elective_rejected",
        "icu_outbound_transfer",
        "icu_inbound_transfer",
        "icu_net_transfer",
    ]

    present_cols = [col for col in candidate_cols if col in summary_df.columns]

    if not present_cols:
        return summary_df.iloc[0:0].copy()

    mask = False
    for col in present_cols:
        mask = mask | (summary_df[col].abs() > tolerance)

    filtered = summary_df[mask].copy()

    sort_cols = ["day", "hospital_id"]
    if "resource" in filtered.columns:
        sort_cols.append("resource")

    filtered = filtered.sort_values(sort_cols).reset_index(drop=True)
    return filtered


def summarize_nonzero_elective_actions(
    solution_tables: Dict[str, pd.DataFrame],
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Return hospital-day rows with nonzero accepted or rejected electives.
    """
    accepted = solution_tables["elective_accepted"].copy()
    rejected = solution_tables["elective_rejected"].copy()

    merged = accepted.merge(
        rejected,
        on=["day", "hospital_id"],
        how="outer",
        validate="one_to_one",
    )

    merged["elective_accepted"] = merged["elective_accepted"].fillna(0.0)
    merged["elective_rejected"] = merged["elective_rejected"].fillna(0.0)

    mask = (
        (merged["elective_accepted"].abs() > tolerance)
        | (merged["elective_rejected"].abs() > tolerance)
    )

    result = merged[mask].sort_values(["day", "hospital_id"]).reset_index(drop=True)
    return result


def summarize_nonzero_transfers(
    solution_tables: Dict[str, pd.DataFrame],
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Return transfer rows with nonzero ICU transfer load.
    """
    transfer_df = solution_tables["icu_transfers"].copy()

    if transfer_df.empty:
        return transfer_df

    mask = transfer_df["icu_transfer_load"].abs() > tolerance
    result = transfer_df[mask].sort_values(
        ["day", "from_hospital", "to_hospital"]
    ).reset_index(drop=True)
    return result


def extract_model_metrics(model: gp.Model) -> pd.DataFrame:
    """
    Extract basic model-level solve metrics as a one-row DataFrame.
    """
    obj_val = None if model.SolCount == 0 else float(model.ObjVal)
    is_mip = bool(model.IsMIP)
    mip_gap = None
    best_bound = None
    if model.SolCount > 0 and is_mip:
        try:
            mip_gap = float(model.MIPGap)
        except Exception:
            mip_gap = None
        try:
            best_bound = float(model.ObjBound)
        except Exception:
            best_bound = None

    metrics = pd.DataFrame(
        [
            {
                "model_status": int(model.Status),
                "solution_count": int(model.SolCount),
                "objective_value": obj_val,
                "best_bound": best_bound,
                "mip_gap": mip_gap,
                "is_mip": is_mip,
                "num_variables": int(model.NumVars),
                "num_constraints": int(model.NumConstrs),
                "runtime_seconds": float(model.Runtime),
            }
        ]
    )

    return metrics