from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.baselines.policy_baselines import build_optimized_network_policy_snapshot
from src.config.dataclasses import HealthcareInstance
from src.markov.forecast_validation import build_capacity_reference_table
from src.simulation.arrival_generator import generate_arrival_scenario
from src.simulation.policy_enforcer import (
    enforce_elective_policy_on_arrivals,
    extract_transfer_policy_table,
    summarize_enforced_arrivals,
)
from src.simulation.twin_engine import simulate_stochastic_twin


# ============================================================
# POLICY SNAPSHOT
# ============================================================

def build_network_policy_snapshot(
    instance: HealthcareInstance,
) -> Dict[str, pd.DataFrame]:
    return build_optimized_network_policy_snapshot(instance)


# ============================================================
# REALIZED CAPACITY
# ============================================================

def evaluate_realized_capacity_pressure(
    realized_resource_demand_df: pd.DataFrame,
    instance: HealthcareInstance,
) -> pd.DataFrame:
    capacity_ref = build_capacity_reference_table(instance)

    merged = realized_resource_demand_df.merge(
        capacity_ref,
        on=["hospital_id", "resource"],
        how="left",
        validate="many_to_one",
    )

    required_capacity_cols = [
        "base_capacity",
        "safe_capacity",
        "max_surge",
        "max_capacity_with_surge",
    ]
    if merged[required_capacity_cols].isnull().any().any():
        raise ValueError("Realized demand rows could not be matched to capacity data.")

    merged["unsafe_excess"] = (
        merged["realized_demand"] - merged["safe_capacity"]
    ).clip(lower=0.0)

    merged["overflow_excess"] = (
        merged["realized_demand"] - merged["base_capacity"]
    ).clip(lower=0.0)

    merged["surge_gap"] = (
        merged["realized_demand"] - merged["max_capacity_with_surge"]
    ).clip(lower=0.0)

    merged["utilization_ratio"] = merged["realized_demand"] / merged["base_capacity"]
    merged["safe_utilization_ratio"] = (
        merged["realized_demand"] / merged["safe_capacity"]
    )

    return merged.reset_index(drop=True)


# ============================================================
# KPI SUMMARY
# ============================================================

def summarize_realized_kpis(
    realized_capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    metrics = {
        "total_unsafe_excess": float(realized_capacity_df["unsafe_excess"].sum()),
        "total_overflow_excess": float(realized_capacity_df["overflow_excess"].sum()),
        "total_surge_gap": float(realized_capacity_df["surge_gap"].sum()),
        "max_utilization_ratio": float(realized_capacity_df["utilization_ratio"].max()),
        "num_unsafe_rows": int((realized_capacity_df["unsafe_excess"] > 0).sum()),
    }

    return pd.DataFrame([metrics])


# ============================================================
# HOSPITAL BOTTLENECK OUTPUTS
# ============================================================

def _extract_transfer_activity_tables(
    transfer_policy_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer_in_df = pd.DataFrame(columns=["hospital_id", "transfer_in"])
    transfer_out_df = pd.DataFrame(columns=["hospital_id", "transfer_out"])

    if transfer_policy_df is None or transfer_policy_df.empty:
        return transfer_in_df, transfer_out_df

    tr = transfer_policy_df.copy()

    possible_cols = [
        "transfer_volume",
        "patients_transferred",
        "transfer_qty",
        "transfer_count",
        "icu_transfer_load",
        "flow",
        "value",
    ]

    volume_col = None
    for c in possible_cols:
        if c in tr.columns:
            volume_col = c
            break

    if volume_col is None:
        tr["transfer_volume"] = 1.0
        volume_col = "transfer_volume"

    tr[volume_col] = pd.to_numeric(tr[volume_col], errors="coerce").fillna(0.0)

    if "from_hospital" in tr.columns:
        transfer_out_df = (
            tr.groupby("from_hospital", as_index=False)[volume_col]
            .sum()
            .rename(columns={
                "from_hospital": "hospital_id",
                volume_col: "transfer_out",
            })
        )

    if "to_hospital" in tr.columns:
        transfer_in_df = (
            tr.groupby("to_hospital", as_index=False)[volume_col]
            .sum()
            .rename(columns={
                "to_hospital": "hospital_id",
                volume_col: "transfer_in",
            })
        )

    return transfer_in_df, transfer_out_df


def _extract_transfer_activity_tables_by_day(
    transfer_policy_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transfer_in_df = pd.DataFrame(columns=["day", "hospital_id", "transfer_in"])
    transfer_out_df = pd.DataFrame(columns=["day", "hospital_id", "transfer_out"])

    if transfer_policy_df is None or transfer_policy_df.empty:
        return transfer_in_df, transfer_out_df

    tr = transfer_policy_df.copy()

    possible_cols = [
        "transfer_volume",
        "patients_transferred",
        "transfer_qty",
        "transfer_count",
        "icu_transfer_load",
        "flow",
        "value",
    ]

    volume_col = None
    for c in possible_cols:
        if c in tr.columns:
            volume_col = c
            break

    if volume_col is None:
        tr["transfer_volume"] = 1.0
        volume_col = "transfer_volume"

    tr[volume_col] = pd.to_numeric(tr[volume_col], errors="coerce").fillna(0.0)

    if "day" not in tr.columns:
        tr["day"] = 0

    if "from_hospital" in tr.columns:
        transfer_out_df = (
            tr.groupby(["day", "from_hospital"], as_index=False)[volume_col]
            .sum()
            .rename(columns={
                "from_hospital": "hospital_id",
                volume_col: "transfer_out",
            })
        )

    if "to_hospital" in tr.columns:
        transfer_in_df = (
            tr.groupby(["day", "to_hospital"], as_index=False)[volume_col]
            .sum()
            .rename(columns={
                "to_hospital": "hospital_id",
                volume_col: "transfer_in",
            })
        )

    return transfer_in_df, transfer_out_df


def build_hospital_bottleneck_outputs(
    realized_capacity_df: pd.DataFrame,
    transfer_policy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if realized_capacity_df.empty:
        return pd.DataFrame(
            columns=[
                "hospital_id",
                "unsafe_excess",
                "overflow_excess",
                "max_utilization",
                "num_unsafe_rows",
                "transfer_in",
                "transfer_out",
                "transfer_net",
                "transfer_activity",
                "bottleneck_score",
            ]
        )

    cap = realized_capacity_df.copy()
    cap["unsafe_flag"] = (cap["unsafe_excess"] > 0).astype(int)

    hospital_summary = (
        cap.groupby("hospital_id", as_index=False)
        .agg(
            unsafe_excess=("unsafe_excess", "sum"),
            overflow_excess=("overflow_excess", "sum"),
            max_utilization=("safe_utilization_ratio", "max"),
            num_unsafe_rows=("unsafe_flag", "sum"),
        )
    )

    transfer_in_df, transfer_out_df = _extract_transfer_activity_tables(transfer_policy_df)

    hospital_summary = hospital_summary.merge(
        transfer_out_df,
        on="hospital_id",
        how="left",
    )
    hospital_summary = hospital_summary.merge(
        transfer_in_df,
        on="hospital_id",
        how="left",
    )

    hospital_summary["transfer_out"] = hospital_summary["transfer_out"].fillna(0.0)
    hospital_summary["transfer_in"] = hospital_summary["transfer_in"].fillna(0.0)

    hospital_summary["transfer_net"] = (
        hospital_summary["transfer_in"] - hospital_summary["transfer_out"]
    )
    hospital_summary["transfer_activity"] = (
        hospital_summary["transfer_in"] + hospital_summary["transfer_out"]
    )

    hospital_summary["bottleneck_score"] = (
        1.0 * hospital_summary["unsafe_excess"]
        + 1.0 * hospital_summary["overflow_excess"]
        + 10.0 * (hospital_summary["max_utilization"] - 1.0).clip(lower=0.0)
        + 0.5 * hospital_summary["num_unsafe_rows"]
        + 0.10 * hospital_summary["transfer_activity"]
    )

    return hospital_summary.sort_values("bottleneck_score", ascending=False).reset_index(drop=True)


# ============================================================
# HOSPITAL TIME-SERIES OUTPUTS
# ============================================================

def build_hospital_time_series_outputs(
    realized_capacity_df: pd.DataFrame,
    transfer_policy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if realized_capacity_df.empty:
        return pd.DataFrame(
            columns=[
                "day",
                "hospital_id",
                "unsafe_excess",
                "overflow_excess",
                "surge_gap",
                "max_utilization",
                "num_unsafe_rows",
                "transfer_in",
                "transfer_out",
                "transfer_net",
                "transfer_activity",
                "bottleneck_score",
            ]
        )

    cap = realized_capacity_df.copy()
    cap["unsafe_flag"] = (cap["unsafe_excess"] > 0).astype(int)

    if "day" not in cap.columns:
        raise ValueError("realized_capacity_df must contain a 'day' column for temporal outputs.")

    hospital_day = (
        cap.groupby(["day", "hospital_id"], as_index=False)
        .agg(
            unsafe_excess=("unsafe_excess", "sum"),
            overflow_excess=("overflow_excess", "sum"),
            surge_gap=("surge_gap", "sum"),
            max_utilization=("safe_utilization_ratio", "max"),
            num_unsafe_rows=("unsafe_flag", "sum"),
        )
    )

    transfer_in_df, transfer_out_df = _extract_transfer_activity_tables_by_day(transfer_policy_df)

    hospital_day = hospital_day.merge(
        transfer_out_df,
        on=["day", "hospital_id"],
        how="left",
    )
    hospital_day = hospital_day.merge(
        transfer_in_df,
        on=["day", "hospital_id"],
        how="left",
    )

    hospital_day["transfer_out"] = hospital_day["transfer_out"].fillna(0.0)
    hospital_day["transfer_in"] = hospital_day["transfer_in"].fillna(0.0)

    hospital_day["transfer_net"] = (
        hospital_day["transfer_in"] - hospital_day["transfer_out"]
    )
    hospital_day["transfer_activity"] = (
        hospital_day["transfer_in"] + hospital_day["transfer_out"]
    )

    hospital_day["bottleneck_score"] = (
        1.0 * hospital_day["unsafe_excess"]
        + 1.0 * hospital_day["overflow_excess"]
        + 10.0 * (hospital_day["max_utilization"] - 1.0).clip(lower=0.0)
        + 0.5 * hospital_day["num_unsafe_rows"]
        + 0.10 * hospital_day["transfer_activity"]
    )

    return hospital_day.sort_values(
        ["day", "bottleneck_score"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ============================================================
# SINGLE RUN
# ============================================================

def run_single_stochastic_evaluation(
    instance: HealthcareInstance,
    policy_snapshot: Optional[Dict[str, pd.DataFrame]] = None,
    arrival_rng: Optional[np.random.Generator] = None,
    transition_rng: Optional[np.random.Generator] = None,
) -> Dict[str, pd.DataFrame]:
    arrival_rng = arrival_rng or np.random.default_rng()
    transition_rng = transition_rng or np.random.default_rng()

    if policy_snapshot is None:
        policy_snapshot = build_network_policy_snapshot(instance)

    realized_arrivals = generate_arrival_scenario(instance, arrival_rng)

    enforced_arrivals = enforce_elective_policy_on_arrivals(
        realized_arrivals, policy_snapshot
    )

    twin_arrivals = enforced_arrivals[
        [
            "day",
            "hospital_id",
            "cohort",
            "arrivals",
            "enforced_realized_arrivals",
        ]
    ].rename(columns={"enforced_realized_arrivals": "realized_arrivals"})

    twin_outputs = simulate_stochastic_twin(
        instance=instance,
        realized_arrivals_df=twin_arrivals,
        rng=transition_rng,
    )

    realized_capacity = evaluate_realized_capacity_pressure(
        twin_outputs["resource_demand"], instance
    )

    transfer_policy = extract_transfer_policy_table(policy_snapshot)
    realized_kpis = summarize_realized_kpis(realized_capacity)
    enforced_arrival_summary = summarize_enforced_arrivals(enforced_arrivals)

    hospital_outputs = build_hospital_bottleneck_outputs(
        realized_capacity_df=realized_capacity,
        transfer_policy_df=transfer_policy,
    )

    hospital_time_series_outputs = build_hospital_time_series_outputs(
        realized_capacity_df=realized_capacity,
        transfer_policy_df=transfer_policy,
    )

    return {
        "policy_name": policy_snapshot["policy_name"],
        "policy_model_metrics": policy_snapshot["model_metrics"],
        "policy_surge": policy_snapshot["surge"],
        "policy_unsafe_slack": policy_snapshot["unsafe_slack"],
        "policy_overflow_slack": policy_snapshot["overflow_slack"],
        "policy_elective_accepted": policy_snapshot["elective_accepted"],
        "policy_elective_rejected": policy_snapshot["elective_rejected"],
        "policy_icu_transfers": transfer_policy,
        "realized_arrivals": realized_arrivals,
        "enforced_arrivals": enforced_arrivals,
        "enforced_arrival_summary": enforced_arrival_summary,
        "state_trajectories": twin_outputs["state_trajectories"],
        "total_state_occupancy": twin_outputs["total_state_occupancy"],
        "resource_demand": twin_outputs["resource_demand"],
        "network_resource_demand": twin_outputs["network_resource_demand"],
        "realized_capacity": realized_capacity,
        "realized_kpis": realized_kpis,
        "hospital_bottleneck_outputs": hospital_outputs,
        "hospital_time_series_outputs": hospital_time_series_outputs,
    }


# ============================================================
# MULTIPLE RUNS
# ============================================================

def run_multiple_stochastic_replications(
    instance: HealthcareInstance,
    policy_snapshot: Optional[Dict[str, pd.DataFrame]] = None,
    n_replications: int = 20,
    base_seed: int = 123,
) -> Dict[str, pd.DataFrame]:
    if policy_snapshot is None:
        policy_snapshot = build_network_policy_snapshot(instance)

    kpi_records = []
    hospital_records = []
    hospital_time_series_records = []

    for rep in range(n_replications):
        outputs = run_single_stochastic_evaluation(
            instance,
            policy_snapshot,
            arrival_rng=np.random.default_rng(base_seed + rep),
            transition_rng=np.random.default_rng(base_seed + 10000 + rep),
        )

        kpi_row = outputs["realized_kpis"].iloc[0].to_dict()
        kpi_row["replication"] = rep + 1

        enforced_summary = outputs["enforced_arrival_summary"]
        kpi_row["total_blocked_arrivals"] = float(
            enforced_summary["arrivals_blocked_by_policy"].sum()
        )

        kpi_records.append(kpi_row)

        h_df = outputs["hospital_bottleneck_outputs"].copy()
        h_df["replication"] = rep + 1
        hospital_records.append(h_df)

        ht_df = outputs["hospital_time_series_outputs"].copy()
        ht_df["replication"] = rep + 1
        hospital_time_series_records.append(ht_df)

    kpi_replications = (
        pd.DataFrame(kpi_records)
        .sort_values("replication")
        .reset_index(drop=True)
    )

    if hospital_records:
        hospital_bottleneck_replications = (
            pd.concat(hospital_records, ignore_index=True)
            .sort_values(["replication", "bottleneck_score"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        hospital_bottleneck_replications = pd.DataFrame()

    if hospital_time_series_records:
        hospital_time_series_replications = (
            pd.concat(hospital_time_series_records, ignore_index=True)
            .sort_values(
                ["replication", "day", "bottleneck_score"],
                ascending=[True, True, False],
            )
            .reset_index(drop=True)
        )
    else:
        hospital_time_series_replications = pd.DataFrame()

    return {
        "kpi_replications": kpi_replications,
        "hospital_bottleneck_replications": hospital_bottleneck_replications,
        "hospital_time_series_replications": hospital_time_series_replications,
    }