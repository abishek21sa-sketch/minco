from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.baselines.policy_baselines import (
    build_local_only_policy_snapshot,
    build_myopic_milp_policy_snapshot,
    build_no_control_policy_snapshot,
    build_no_transfer_policy_snapshot,
    build_optimized_network_policy_snapshot,
)
from src.config.dataclasses import CapacitiesData, HealthcareInstance, TransferLanesData
from src.config.load_parameters import apply_parameter_config
from src.config.loader import load_healthcare_instance
from src.optimization.robust_minco import (
    build_robust_optimized_network_policy_snapshot,
)
from src.optimization.robust_uncertainty import RobustArrivalConfig
from src.simulation.policy_runner import run_multiple_stochastic_replications


ROBUST_CONFIG = RobustArrivalConfig(
    default_rho=0.10,
    rho_by_hospital={"H3": 0.20},
    scenario_multiplier=1.05,
)


def clone_instance_with_reduced_h3_icu_capacity(
    instance: HealthcareInstance,
    new_capacity: float,
) -> HealthcareInstance:
    capacities_df = instance.capacities.df.copy()
    hospital_col = "hospital_id" if "hospital_id" in capacities_df.columns else "hospital"
    resource_col = "resource" if "resource" in capacities_df.columns else "unit"
    capacity_col = "base_capacity" if "base_capacity" in capacities_df.columns else "capacity"

    capacities_df[capacity_col] = capacities_df[capacity_col].astype(float)

    mask = (
        (capacities_df[hospital_col] == "H3")
        & (capacities_df[resource_col].astype(str).str.upper() == "ICU")
    )
    capacities_df.loc[mask, capacity_col] = float(new_capacity)

    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=CapacitiesData(df=capacities_df),
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=instance.transfer_lanes,
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=instance.arrivals,
        transitions=instance.transitions,
    )


def clone_instance_with_transfers_disabled(
    instance: HealthcareInstance,
) -> HealthcareInstance:
    transfer_df = instance.transfer_lanes.df.copy()

    if "allowed" in transfer_df.columns:
        transfer_df["allowed"] = 0
    elif "capacity" in transfer_df.columns:
        transfer_df["capacity"] = 0
    elif "max_transfer" in transfer_df.columns:
        transfer_df["max_transfer"] = 0
    elif "transfer_capacity" in transfer_df.columns:
        transfer_df["transfer_capacity"] = 0
    else:
        transfer_df["allowed"] = 0

    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=instance.capacities,
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=TransferLanesData(df=transfer_df),
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=instance.arrivals,
        transitions=instance.transitions,
    )


def summarize_policy_replications(
    replications_df: pd.DataFrame,
    policy_name: str,
    scenario_name: str,
) -> pd.DataFrame:
    summary = {
        "policy_name": policy_name,
        "scenario": scenario_name,
        "n_replications": int(len(replications_df)),
    }

    metric_cols = [
        "total_unsafe_excess",
        "total_overflow_excess",
        "total_surge_gap",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
    ]

    metric_cols = [c for c in metric_cols if c in replications_df.columns]

    for col in metric_cols:
        summary[f"{col}_mean"] = float(replications_df[col].mean())
        summary[f"{col}_std"] = (
            float(replications_df[col].std(ddof=1))
            if len(replications_df) > 1
            else 0.0
        )
        summary[f"{col}_min"] = float(replications_df[col].min())
        summary[f"{col}_max"] = float(replications_df[col].max())

    return pd.DataFrame([summary])


def summarize_hospital_bottlenecks(
    hospital_replications: pd.DataFrame,
) -> pd.DataFrame:
    if hospital_replications.empty:
        return pd.DataFrame()

    summary = (
        hospital_replications.groupby(
            ["scenario", "policy_name", "hospital_id"],
            as_index=False,
        )
        .agg(
            unsafe_excess_mean=("unsafe_excess", "mean"),
            unsafe_excess_std=("unsafe_excess", "std"),
            overflow_excess_mean=("overflow_excess", "mean"),
            overflow_excess_std=("overflow_excess", "std"),
            max_utilization_mean=("max_utilization", "mean"),
            max_utilization_std=("max_utilization", "std"),
            num_unsafe_rows_mean=("num_unsafe_rows", "mean"),
            num_unsafe_rows_std=("num_unsafe_rows", "std"),
            transfer_in_mean=("transfer_in", "mean"),
            transfer_out_mean=("transfer_out", "mean"),
            transfer_net_mean=("transfer_net", "mean"),
            transfer_activity_mean=("transfer_activity", "mean"),
            bottleneck_score_mean=("bottleneck_score", "mean"),
            bottleneck_score_std=("bottleneck_score", "std"),
        )
    )

    std_cols = [c for c in summary.columns if c.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0.0)

    return summary.sort_values(
        ["scenario", "policy_name", "bottleneck_score_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def decompose_bottleneck(hospital_replications: pd.DataFrame) -> pd.DataFrame:
    if hospital_replications.empty:
        return pd.DataFrame()

    df = hospital_replications.copy()

    df["util_component_mean_proxy"] = 10.0 * (df["max_utilization"] - 1.0).clip(lower=0.0)
    df["unsafe_component_mean_proxy"] = 1.0 * df["unsafe_excess"]
    df["overflow_component_mean_proxy"] = 1.0 * df["overflow_excess"]
    df["event_component_mean_proxy"] = 0.5 * df["num_unsafe_rows"]
    df["transfer_component_mean_proxy"] = 0.10 * df["transfer_activity"]

    grouped = (
        df.groupby(["scenario", "policy_name", "hospital_id"], as_index=False)
        .agg(
            util_component_mean=("util_component_mean_proxy", "mean"),
            unsafe_component_mean=("unsafe_component_mean_proxy", "mean"),
            overflow_component_mean=("overflow_component_mean_proxy", "mean"),
            event_component_mean=("event_component_mean_proxy", "mean"),
            transfer_component_mean=("transfer_component_mean_proxy", "mean"),
            bottleneck_score_mean=("bottleneck_score", "mean"),
        )
    )

    return grouped.sort_values(
        ["scenario", "policy_name", "bottleneck_score_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def summarize_hospital_time_series(
    hospital_time_series_df: pd.DataFrame,
) -> pd.DataFrame:
    if hospital_time_series_df.empty:
        return pd.DataFrame()

    working = hospital_time_series_df.copy()

    peak_idx = (
        working.groupby(["scenario", "policy_name", "hospital_id"])["bottleneck_score"]
        .idxmax()
    )
    peak_rows = (
        working.loc[peak_idx, ["scenario", "policy_name", "hospital_id", "day", "bottleneck_score"]]
        .rename(columns={
            "day": "peak_bottleneck_day",
            "bottleneck_score": "peak_bottleneck_score",
        })
        .reset_index(drop=True)
    )

    summary = (
        working.groupby(
            ["scenario", "policy_name", "hospital_id"],
            as_index=False,
        )
        .agg(
            mean_daily_bottleneck_score=("bottleneck_score", "mean"),
            max_daily_bottleneck_score=("bottleneck_score", "max"),
            mean_daily_unsafe_excess=("unsafe_excess", "mean"),
            max_daily_unsafe_excess=("unsafe_excess", "max"),
            mean_daily_overflow_excess=("overflow_excess", "mean"),
            max_daily_overflow_excess=("overflow_excess", "max"),
            mean_daily_surge_gap=("surge_gap", "mean"),
            max_daily_surge_gap=("surge_gap", "max"),
            mean_daily_utilization=("max_utilization", "mean"),
            max_daily_utilization=("max_utilization", "max"),
            mean_daily_transfer_activity=("transfer_activity", "mean"),
            max_daily_transfer_activity=("transfer_activity", "max"),
            days_with_unsafe_excess=("unsafe_excess", lambda s: int((s > 0).sum())),
            days_with_overflow=("overflow_excess", lambda s: int((s > 0).sum())),
            days_above_safe_utilization=("max_utilization", lambda s: int((s > 1.0).sum())),
        )
    )

    summary = summary.merge(
        peak_rows,
        on=["scenario", "policy_name", "hospital_id"],
        how="left",
    )

    return summary.sort_values(
        ["scenario", "policy_name", "max_daily_bottleneck_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _extract_outputs(outputs: Any, policy_name: str) -> dict[str, pd.DataFrame]:
    if not isinstance(outputs, dict):
        raise TypeError(
            f"Expected run_multiple_stochastic_replications(...) to return dict, got {type(outputs)}"
        )

    required = [
        "kpi_replications",
        "hospital_bottleneck_replications",
        "hospital_time_series_replications",
    ]
    missing = [k for k in required if k not in outputs]
    if missing:
        raise ValueError(
            f"Missing expected output keys for policy '{policy_name}': {missing}. "
            f"Found keys: {list(outputs.keys())}"
        )

    kpi_replications = outputs["kpi_replications"].copy()
    kpi_replications["policy_name"] = policy_name

    hospital_replications = outputs["hospital_bottleneck_replications"].copy()
    hospital_replications["policy_name"] = policy_name

    hospital_time_series = outputs["hospital_time_series_replications"].copy()
    hospital_time_series["policy_name"] = policy_name

    return {
        "kpi_replications": kpi_replications,
        "hospital_bottleneck_replications": hospital_replications,
        "hospital_time_series_replications": hospital_time_series,
    }


def run_policy_replications_for_instance(
    instance: HealthcareInstance,
    n_replications: int = 50,
    base_seed: int = 123,
) -> dict[str, dict[str, pd.DataFrame]]:
    policy_builders: dict[str, Callable[[HealthcareInstance], Any]] = {
        "optimized_network": build_optimized_network_policy_snapshot,
        "robust_optimized_network": lambda inst: build_robust_optimized_network_policy_snapshot(
            inst,
            robust_config=ROBUST_CONFIG,
        ),
        "no_transfer": build_no_transfer_policy_snapshot,
        "no_control": build_no_control_policy_snapshot,
        "local_only": build_local_only_policy_snapshot,
        "myopic_milp": build_myopic_milp_policy_snapshot,
    }

    outputs_by_policy: dict[str, dict[str, pd.DataFrame]] = {}

    for offset, (policy_name, builder) in enumerate(policy_builders.items()):
        policy_snapshot = builder(instance)

        outputs = run_multiple_stochastic_replications(
            instance=instance,
            policy_snapshot=policy_snapshot,
            n_replications=n_replications,
            base_seed=base_seed + 1000 * offset,
        )

        outputs_by_policy[policy_name] = _extract_outputs(outputs, policy_name)

    return outputs_by_policy


def run_scenario(
    scenario_name: str,
    instance: HealthcareInstance,
    n_replications: int = 50,
    base_seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs_by_policy = run_policy_replications_for_instance(
        instance=instance,
        n_replications=n_replications,
        base_seed=base_seed,
    )

    all_replications = []
    all_summaries = []
    all_hospital_replications = []
    all_hospital_time_series = []

    for policy_name, outputs in outputs_by_policy.items():
        replications_df = outputs["kpi_replications"].copy()
        replications_df["scenario"] = scenario_name
        all_replications.append(replications_df)

        summary_df = summarize_policy_replications(
            replications_df=replications_df,
            policy_name=policy_name,
            scenario_name=scenario_name,
        )
        all_summaries.append(summary_df)

        hospital_df = outputs["hospital_bottleneck_replications"].copy()
        hospital_df["scenario"] = scenario_name
        all_hospital_replications.append(hospital_df)

        hospital_ts_df = outputs["hospital_time_series_replications"].copy()
        hospital_ts_df["scenario"] = scenario_name
        all_hospital_time_series.append(hospital_ts_df)

    stacked_replications = pd.concat(all_replications, ignore_index=True)
    stacked_summaries = pd.concat(all_summaries, ignore_index=True)
    stacked_hospital_replications = pd.concat(all_hospital_replications, ignore_index=True)
    stacked_hospital_time_series = pd.concat(all_hospital_time_series, ignore_index=True)

    stacked_replications = stacked_replications.sort_values(
        ["scenario", "policy_name", "replication"]
    ).reset_index(drop=True)

    stacked_summaries = stacked_summaries.sort_values(
        ["scenario", "policy_name"]
    ).reset_index(drop=True)

    stacked_hospital_replications = stacked_hospital_replications.sort_values(
        ["scenario", "policy_name", "replication", "bottleneck_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    stacked_hospital_time_series = stacked_hospital_time_series.sort_values(
        ["scenario", "policy_name", "replication", "day", "bottleneck_score"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    return (
        stacked_replications,
        stacked_summaries,
        stacked_hospital_replications,
        stacked_hospital_time_series,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic.json",
        help="Path to parameter config JSON file.",
    )
    parser.add_argument(
        "--n_replications",
        type=int,
        default=50,
        help="Number of stochastic replications per policy/scenario.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_instance = load_healthcare_instance()
    base_instance = apply_parameter_config(base_instance, args.config)

    config_stem = Path(args.config).stem
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_results = []

    scenario_results.append(
        run_scenario(
            scenario_name="baseline",
            instance=base_instance,
            n_replications=args.n_replications,
            base_seed=123,
        )
    )

    reduced_h3_instance = clone_instance_with_reduced_h3_icu_capacity(
        instance=base_instance,
        new_capacity=6.0,
    )
    scenario_results.append(
        run_scenario(
            scenario_name="h3_icu_capacity_reduced",
            instance=reduced_h3_instance,
            n_replications=args.n_replications,
            base_seed=423,
        )
    )

    no_transfer_instance = clone_instance_with_transfers_disabled(
        instance=base_instance,
    )
    scenario_results.append(
        run_scenario(
            scenario_name="transfer_disabled",
            instance=no_transfer_instance,
            n_replications=args.n_replications,
            base_seed=523,
        )
    )

    all_replications = pd.concat(
        [rep for rep, _, _, _ in scenario_results],
        ignore_index=True,
    )
    all_summaries = pd.concat(
        [summ for _, summ, _, _ in scenario_results],
        ignore_index=True,
    )
    all_hospital_replications = pd.concat(
        [hosp for _, _, hosp, _ in scenario_results],
        ignore_index=True,
    )
    all_hospital_time_series = pd.concat(
        [hts for _, _, _, hts in scenario_results],
        ignore_index=True,
    )

    all_replications = all_replications.sort_values(
        ["scenario", "policy_name", "replication"]
    ).reset_index(drop=True)

    all_summaries = all_summaries.sort_values(
        ["scenario", "policy_name"]
    ).reset_index(drop=True)

    all_hospital_replications = all_hospital_replications.sort_values(
        ["scenario", "policy_name", "replication", "bottleneck_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    all_hospital_time_series = all_hospital_time_series.sort_values(
        ["scenario", "policy_name", "replication", "day", "bottleneck_score"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    hospital_summary = summarize_hospital_bottlenecks(all_hospital_replications)
    decomp = decompose_bottleneck(all_hospital_replications)
    hospital_time_series_summary = summarize_hospital_time_series(all_hospital_time_series)

    replications_path = output_dir / f"scenario_suite_replications_{config_stem}.csv"
    summaries_path = output_dir / f"scenario_suite_summary_{config_stem}.csv"
    hospital_replications_path = output_dir / f"hospital_bottleneck_replications_{config_stem}.csv"
    hospital_summary_path = output_dir / f"hospital_bottleneck_summary_{config_stem}.csv"
    hospital_decomposition_path = output_dir / f"hospital_bottleneck_decomposition_{config_stem}.csv"
    hospital_time_series_path = output_dir / f"hospital_time_series_outputs_{config_stem}.csv"
    hospital_time_series_summary_path = output_dir / f"hospital_time_series_summary_{config_stem}.csv"

    all_replications.to_csv(replications_path, index=False)
    all_summaries.to_csv(summaries_path, index=False)
    all_hospital_replications.to_csv(hospital_replications_path, index=False)
    hospital_summary.to_csv(hospital_summary_path, index=False)
    decomp.to_csv(hospital_decomposition_path, index=False)
    all_hospital_time_series.to_csv(hospital_time_series_path, index=False)
    hospital_time_series_summary.to_csv(hospital_time_series_summary_path, index=False)

    print(f"\nCONFIG MODE: {config_stem}")

    print("\nSCENARIO SUITE SUMMARY")
    print(all_summaries)

    print("\nHOSPITAL BOTTLENECK SUMMARY")
    print(hospital_summary)

    print("\nHOSPITAL BOTTLENECK DECOMPOSITION")
    print(decomp)

    print("\nHOSPITAL TIME SERIES SUMMARY")
    print(hospital_time_series_summary)

    print("\nSaved files:")
    print(replications_path)
    print(summaries_path)
    print(hospital_replications_path)
    print(hospital_summary_path)
    print(hospital_decomposition_path)
    print(hospital_time_series_path)
    print(hospital_time_series_summary_path)


if __name__ == "__main__":
    main()