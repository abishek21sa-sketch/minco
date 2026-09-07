from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.baselines.policy_baselines import (
    build_local_only_policy_snapshot,
    build_no_control_policy_snapshot,
    build_no_transfer_policy_snapshot,
    build_optimized_network_policy_snapshot,
    build_myopic_milp_policy_snapshot,
)
from src.config.loader import load_healthcare_instance
from src.simulation.policy_runner import run_multiple_stochastic_replications


def summarize_policy_replications(
    replications_df: pd.DataFrame,
    policy_name: str,
    scenario_name: str,
) -> pd.DataFrame:
    """
    Summarize one policy-scenario replication table into a one-row summary.
    """
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
        "max_safe_utilization_ratio",
        "num_unsafe_rows",
        "num_overflow_rows",
        "num_surge_gap_rows",
    ]

    for col in metric_cols:
        summary[f"{col}_mean"] = float(replications_df[col].mean())
        summary[f"{col}_std"] = float(replications_df[col].std(ddof=1)) if len(replications_df) > 1 else 0.0
        summary[f"{col}_min"] = float(replications_df[col].min())
        summary[f"{col}_max"] = float(replications_df[col].max())

    return pd.DataFrame([summary])


def run_policy_replications(
    n_replications: int = 50,
    base_seed: int = 123,
    arrival_shock_multiplier: float = 1.0,
    shock_cohort: str | None = None,
    shock_hospital_id: str | None = None,
    shock_start_day: int | None = None,
    shock_end_day: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run all current policy baselines under one scenario.

    NOTE:
    The current simulation layer is open-loop, so these policy snapshots are
    generated and stored here for experiment bookkeeping, but realized stochastic
    KPIs come from the shared stochastic evaluation pipeline.
    """
    instance = load_healthcare_instance()

    policy_builders: dict[str, Callable] = {
        "optimized_network": build_optimized_network_policy_snapshot,
        "no_transfer": build_no_transfer_policy_snapshot,
        "no_control": build_no_control_policy_snapshot,
        "local_only": build_local_only_policy_snapshot,
        "myopic_milp": build_myopic_milp_policy_snapshot,
    }

    replication_tables: dict[str, pd.DataFrame] = {}

    for offset, (policy_name, builder) in enumerate(policy_builders.items()):
        _ = builder(instance)

        replications = run_multiple_stochastic_replications(
            instance=instance,
            n_replications=n_replications,
            base_seed=base_seed + 1000 * offset,
            arrival_shock_multiplier=arrival_shock_multiplier,
            shock_cohort=shock_cohort,
            shock_hospital_id=shock_hospital_id,
            shock_start_day=shock_start_day,
            shock_end_day=shock_end_day,
        )

        replications = replications.copy()
        replications["policy_name"] = policy_name
        replication_tables[policy_name] = replications

    return replication_tables


def run_policy_comparison_for_scenario(
    scenario_name: str,
    n_replications: int = 50,
    base_seed: int = 123,
    arrival_shock_multiplier: float = 1.0,
    shock_cohort: str | None = None,
    shock_hospital_id: str | None = None,
    shock_start_day: int | None = None,
    shock_end_day: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all policies for one scenario and return:
    - stacked replication table
    - stacked summary table
    """
    replication_tables = run_policy_replications(
        n_replications=n_replications,
        base_seed=base_seed,
        arrival_shock_multiplier=arrival_shock_multiplier,
        shock_cohort=shock_cohort,
        shock_hospital_id=shock_hospital_id,
        shock_start_day=shock_start_day,
        shock_end_day=shock_end_day,
    )

    all_replications = []
    all_summaries = []

    for policy_name, replications_df in replication_tables.items():
        replications = replications_df.copy()
        replications["scenario"] = scenario_name
        all_replications.append(replications)

        summary_df = summarize_policy_replications(
            replications_df=replications_df,
            policy_name=policy_name,
            scenario_name=scenario_name,
        )
        all_summaries.append(summary_df)

    stacked_replications = pd.concat(all_replications, ignore_index=True)
    stacked_summaries = pd.concat(all_summaries, ignore_index=True)

    stacked_replications = stacked_replications.sort_values(
        ["policy_name", "replication"]
    ).reset_index(drop=True)

    stacked_summaries = stacked_summaries.sort_values(
        ["scenario", "policy_name"]
    ).reset_index(drop=True)

    return stacked_replications, stacked_summaries


def main() -> None:
    """
    Run baseline and shock policy-comparison studies.
    """
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_replications, baseline_summary = run_policy_comparison_for_scenario(
        scenario_name="baseline",
        n_replications=50,
        base_seed=123,
    )

    shock_replications, shock_summary = run_policy_comparison_for_scenario(
        scenario_name="h3_c4_shock",
        n_replications=50,
        base_seed=123,
        arrival_shock_multiplier=1.5,
        shock_cohort="c4",
        shock_hospital_id="H3",
        shock_start_day=5,
        shock_end_day=7,
    )

    combined_replications = pd.concat(
        [baseline_replications, shock_replications],
        ignore_index=True,
    )
    combined_summary = pd.concat(
        [baseline_summary, shock_summary],
        ignore_index=True,
    )

    baseline_replications.to_csv(
        output_dir / "policy_comparison_baseline_replications.csv",
        index=False,
    )
    baseline_summary.to_csv(
        output_dir / "policy_comparison_baseline_summary.csv",
        index=False,
    )

    shock_replications.to_csv(
        output_dir / "policy_comparison_h3_c4_shock_replications.csv",
        index=False,
    )
    shock_summary.to_csv(
        output_dir / "policy_comparison_h3_c4_shock_summary.csv",
        index=False,
    )

    combined_replications.to_csv(
        output_dir / "policy_comparison_combined_replications.csv",
        index=False,
    )
    combined_summary.to_csv(
        output_dir / "policy_comparison_combined_summary.csv",
        index=False,
    )

    print("\nBASELINE POLICY SUMMARY")
    print(baseline_summary)

    print("\nH3 C4 SHOCK POLICY SUMMARY")
    print(shock_summary)

    print("\nCOMBINED POLICY SUMMARY")
    print(combined_summary)

    print("\nSaved files:")
    print(output_dir / "policy_comparison_baseline_replications.csv")
    print(output_dir / "policy_comparison_baseline_summary.csv")
    print(output_dir / "policy_comparison_h3_c4_shock_replications.csv")
    print(output_dir / "policy_comparison_h3_c4_shock_summary.csv")
    print(output_dir / "policy_comparison_combined_replications.csv")
    print(output_dir / "policy_comparison_combined_summary.csv")


if __name__ == "__main__":
    main()