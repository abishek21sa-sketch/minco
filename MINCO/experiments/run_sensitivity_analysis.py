from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config.dataclasses import CostsData, HealthcareInstance
from src.config.loader import load_healthcare_instance
from src.optimizer.model_builder import solve_network_capacity_model
from src.optimizer.solution_parser import (
    extract_model_metrics,
    extract_network_capacity_solution,
    summarize_nonzero_elective_actions,
    summarize_nonzero_transfers,
)


def clone_instance_with_updated_costs(
    instance: HealthcareInstance,
    elective_rejection_cost: float | None = None,
    generic_transfer_cost: float | None = None,
) -> HealthcareInstance:
    """
    Return a copy of the instance with selected costs overwritten in memory.
    """
    costs_df = instance.costs.df.copy()

    if elective_rejection_cost is not None:
        costs_df.loc[
            costs_df["cost_name"] == "elective_rejection",
            "value",
        ] = float(elective_rejection_cost)

    if generic_transfer_cost is not None:
        costs_df.loc[
            costs_df["cost_name"] == "transfer",
            "value",
        ] = float(generic_transfer_cost)

    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=instance.capacities,
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=instance.transfer_lanes,
        costs=CostsData(df=costs_df),
        elective_bounds=instance.elective_bounds,
        arrivals=instance.arrivals,
        transitions=instance.transitions,
    )


def summarize_solution_behavior(
    solution_tables: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """
    Aggregate high-level behavior from solved network-model outputs.
    """
    elective_df = summarize_nonzero_elective_actions(solution_tables)
    transfer_df = summarize_nonzero_transfers(solution_tables)

    total_elective_accepted = 0.0
    total_elective_rejected = 0.0
    if not elective_df.empty:
        total_elective_accepted = float(elective_df["elective_accepted"].sum())
        total_elective_rejected = float(elective_df["elective_rejected"].sum())

    total_icu_transfer = 0.0
    total_transfer_cost_incurred = 0.0
    if not transfer_df.empty:
        total_icu_transfer = float(transfer_df["icu_transfer_load"].sum())
        if "transfer_cost_incurred" in transfer_df.columns:
            total_transfer_cost_incurred = float(
                transfer_df["transfer_cost_incurred"].sum()
            )

    return {
        "total_elective_accepted": total_elective_accepted,
        "total_elective_rejected": total_elective_rejected,
        "total_icu_transfer": total_icu_transfer,
        "total_transfer_cost_incurred": total_transfer_cost_incurred,
        "num_transfer_rows": float(len(transfer_df)),
        "num_elective_rows": float(len(elective_df)),
    }


def run_single_sensitivity_case(
    base_instance: HealthcareInstance,
    elective_rejection_cost: float,
    generic_transfer_cost: float,
) -> dict[str, float]:
    """
    Solve one sensitivity case and return summarized metrics.
    """
    instance = clone_instance_with_updated_costs(
        instance=base_instance,
        elective_rejection_cost=elective_rejection_cost,
        generic_transfer_cost=generic_transfer_cost,
    )

    model, variables, validated, elective_response, arcs, transfer_cost_lookup = (
        solve_network_capacity_model(
            instance=instance,
            output_flag=0,
        )
    )

    solution_tables = extract_network_capacity_solution(
        model=model,
        variables=variables,
        transfer_cost_lookup=transfer_cost_lookup,
    )

    model_metrics = extract_model_metrics(model).iloc[0].to_dict()
    behavior_metrics = summarize_solution_behavior(solution_tables)

    return {
        "elective_rejection_cost": float(elective_rejection_cost),
        "generic_transfer_cost": float(generic_transfer_cost),
        **model_metrics,
        **behavior_metrics,
    }


def run_sensitivity_grid(
    elective_rejection_costs: Iterable[float],
    generic_transfer_costs: Iterable[float],
) -> pd.DataFrame:
    """
    Run the full 2D sensitivity grid and return results as a DataFrame.
    """
    base_instance = load_healthcare_instance()

    records: list[dict[str, float]] = []

    for rejection_cost in elective_rejection_costs:
        for transfer_cost in generic_transfer_costs:
            result = run_single_sensitivity_case(
                base_instance=base_instance,
                elective_rejection_cost=float(rejection_cost),
                generic_transfer_cost=float(transfer_cost),
            )
            records.append(result)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(
        ["elective_rejection_cost", "generic_transfer_cost"]
    ).reset_index(drop=True)

    return results_df


def main() -> None:
    """
    Run a starter sensitivity study and save results.
    """
    rejection_grid = [10, 20, 30, 40, 60]
    transfer_grid = [0, 10, 20, 40, 60, 80, 100]

    results_df = run_sensitivity_grid(
        elective_rejection_costs=rejection_grid,
        generic_transfer_costs=transfer_grid,
    )

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "sensitivity_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nSENSITIVITY RESULTS")
    print(results_df)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()