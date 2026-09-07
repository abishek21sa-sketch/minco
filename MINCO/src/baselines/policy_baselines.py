from __future__ import annotations

from typing import Dict

import pandas as pd
import gurobipy as gp

from src.config.dataclasses import HealthcareInstance
from src.optimizer.model_builder import (
    solve_extended_capacity_model,
    solve_network_capacity_model,
)
from src.optimizer.solution_parser import (
    extract_extended_capacity_solution,
    extract_network_capacity_solution,
    extract_model_metrics,
)
from src.optimizer.constraints import add_network_capacity_constraints
from src.optimizer.objective import set_network_capacity_objective
from src.optimizer.variables import add_network_capacity_response_variables
from src.optimizer.transfer_utils import build_transfer_capacity_lookup


def _zero_out_transfer_table(transfer_df: pd.DataFrame) -> pd.DataFrame:
    out = transfer_df.copy()
    if "icu_transfer_load" in out.columns:
        out["icu_transfer_load"] = 0.0
    if "transfer_cost_incurred" in out.columns:
        out["transfer_cost_incurred"] = 0.0
    return out


def _force_accept_all_electives(
    accepted_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = accepted_df.copy()
    rejected = rejected_df.copy()

    merged = accepted.merge(
        rejected,
        on=["day", "hospital_id"],
        how="outer",
        validate="one_to_one",
        suffixes=("_accepted", "_rejected"),
    )

    merged["elective_accepted"] = (
        merged["elective_accepted"].fillna(0.0)
        + merged["elective_rejected"].fillna(0.0)
    )
    merged["elective_rejected"] = 0.0

    accepted_out = merged[["day", "hospital_id", "elective_accepted"]].copy()
    rejected_out = merged[["day", "hospital_id", "elective_rejected"]].copy()

    return (
        accepted_out.sort_values(["day", "hospital_id"]).reset_index(drop=True),
        rejected_out.sort_values(["day", "hospital_id"]).reset_index(drop=True),
    )


# ============================================================
# OPTIMIZED NETWORK (MINCO)
# ============================================================

def build_optimized_network_policy_snapshot(
    instance: HealthcareInstance,
) -> Dict[str, pd.DataFrame]:

    model, variables, validated, elective_response, arcs, transfer_cost_lookup = (
        solve_network_capacity_model(instance=instance, output_flag=0)
    )

    solution_tables = extract_network_capacity_solution(
        model=model,
        variables=variables,
        transfer_cost_lookup=transfer_cost_lookup,
    )

    return {
        "policy_name": pd.DataFrame([{"policy_name": "optimized_network"}]),
        "model_metrics": extract_model_metrics(model),
        **solution_tables,
    }


# ============================================================
# NO TRANSFER
# ============================================================

def build_no_transfer_policy_snapshot(
    instance: HealthcareInstance,
) -> Dict[str, pd.DataFrame]:

    snapshot = build_optimized_network_policy_snapshot(instance)
    snapshot["icu_transfers"] = _zero_out_transfer_table(snapshot["icu_transfers"])
    snapshot["policy_name"] = pd.DataFrame([{"policy_name": "no_transfer"}])

    return snapshot


# ============================================================
# NO CONTROL
# ============================================================

def build_no_control_policy_snapshot(
    instance: HealthcareInstance,
) -> Dict[str, pd.DataFrame]:

    model, variables, validated, elective_response = solve_extended_capacity_model(
        instance=instance,
        output_flag=0,
    )

    solution_tables = extract_extended_capacity_solution(model=model, variables=variables)

    accepted_all, rejected_zero = _force_accept_all_electives(
        solution_tables["elective_accepted"],
        solution_tables["elective_rejected"],
    )

    empty_transfer = pd.DataFrame(
        columns=["from_hospital", "to_hospital", "day", "icu_transfer_load"]
    )

    return {
        "policy_name": pd.DataFrame([{"policy_name": "no_control"}]),
        "model_metrics": extract_model_metrics(model),
        "surge": solution_tables["surge"],
        "unsafe_slack": solution_tables["unsafe_slack"],
        "overflow_slack": solution_tables["overflow_slack"],
        "elective_accepted": accepted_all,
        "elective_rejected": rejected_zero,
        "icu_transfers": empty_transfer,
    }


# ============================================================
# LOCAL ONLY
# ============================================================

def build_local_only_policy_snapshot(
    instance: HealthcareInstance,
) -> Dict[str, pd.DataFrame]:

    snapshot = build_no_control_policy_snapshot(instance)
    snapshot["policy_name"] = pd.DataFrame([{"policy_name": "local_only"}])
    return snapshot


# ============================================================
# MYOPIC MILP (FIXED VERSION)
# ============================================================

def build_myopic_milp_policy_snapshot(
    instance: HealthcareInstance,
) -> Dict[str, pd.DataFrame]:

    model, variables, validated, elective_response, arcs, transfer_cost_lookup = (
        solve_network_capacity_model(instance=instance, output_flag=0)
    )

    solution_tables = extract_network_capacity_solution(
        model=model,
        variables=variables,
        transfer_cost_lookup=transfer_cost_lookup,
    )

    days = sorted(validated["day"].unique().tolist())
    transfer_capacity_lookup = build_transfer_capacity_lookup(instance)

    def zero_numeric_except_day(df):
        out = df.copy()
        num_cols = out.select_dtypes(include=["number"]).columns.tolist()
        num_cols = [c for c in num_cols if c != "day"]
        out.loc[:, num_cols] = 0.0
        return out

    surge_out = zero_numeric_except_day(solution_tables["surge"])
    unsafe_out = zero_numeric_except_day(solution_tables["unsafe_slack"])
    overflow_out = zero_numeric_except_day(solution_tables["overflow_slack"])
    accepted_out = zero_numeric_except_day(solution_tables["elective_accepted"])
    rejected_out = zero_numeric_except_day(solution_tables["elective_rejected"])
    transfers_out = zero_numeric_except_day(solution_tables["icu_transfers"])

    for day in days:
        day_validated = validated[validated["day"] == day].copy()
        if day_validated.empty:
            continue

        day_model = gp.Model(f"myopic_day_{day}")

        hospitals = sorted(day_validated["hospital_id"].unique())
        resources = sorted(day_validated["resource"].unique())

        day_vars = add_network_capacity_response_variables(
            model=day_model,
            days=[day],
            hospitals=hospitals,
            resources=resources,
            arcs=arcs,
        )

        bounds = instance.elective_bounds.df[
            instance.elective_bounds.df["day"] == day
        ].copy()

        response = elective_response[
            (elective_response["day"] == day)
            & (elective_response["admit_day"] == day)
        ].copy()

        set_network_capacity_objective(
            model=day_model,
            variables=day_vars,
            validated_forecast_df=day_validated,
            elective_bounds_df=bounds,
            transfer_cost_lookup=transfer_cost_lookup,
            days=[day],
            costs_df=instance.costs.df,
        )

        add_network_capacity_constraints(
            model=day_model,
            variables=day_vars,
            validated_forecast_df=day_validated,
            elective_bounds_df=bounds,
            elective_response_df=response,
            arcs=arcs,
            transfer_capacity_lookup=transfer_capacity_lookup,
        )

        day_model.setParam("OutputFlag", 0)
        day_model.optimize()

        if day_model.SolCount == 0:
            continue

        day_tables = extract_network_capacity_solution(
            model=day_model,
            variables=day_vars,
            transfer_cost_lookup=transfer_cost_lookup,
        )

        for src_df, dst_df, value_col, keys in [
            (day_tables["surge"], surge_out, "surge_activated", ["hospital_id", "resource", "day"]),
            (day_tables["unsafe_slack"], unsafe_out, "unsafe_slack", ["hospital_id", "resource", "day"]),
            (day_tables["overflow_slack"], overflow_out, "overflow_slack", ["hospital_id", "resource", "day"]),
            (day_tables["elective_accepted"], accepted_out, "elective_accepted", ["hospital_id", "day"]),
            (day_tables["elective_rejected"], rejected_out, "elective_rejected", ["hospital_id", "day"]),
            (day_tables["icu_transfers"], transfers_out, "icu_transfer_load", ["from_hospital", "to_hospital", "day"]),
        ]:

            if src_df is None or src_df.empty:
                continue

            if not all(col in src_df.columns for col in keys + [value_col]):
                continue

            lookup = src_df.set_index(keys)[value_col].to_dict()

            for idx, row in dst_df.iterrows():
                key = tuple(row[k] for k in keys)
                if key in lookup:
                    dst_df.at[idx, value_col] = float(lookup[key])

    return {
        "policy_name": pd.DataFrame([{"policy_name": "myopic_milp"}]),
        "model_metrics": extract_model_metrics(model),
        "surge": surge_out,
        "unsafe_slack": unsafe_out,
        "overflow_slack": overflow_out,
        "elective_accepted": accepted_out,
        "elective_rejected": rejected_out,
        "icu_transfers": transfers_out,
    }