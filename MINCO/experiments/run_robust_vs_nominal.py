from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.loader import load_healthcare_instance
from src.config.dataclasses import HealthcareInstance
from src.optimization.robust_minco import (
    build_robust_optimized_network_policy_snapshot,
)
from src.optimization.robust_uncertainty import (
    RobustArrivalConfig,
)
from src.baselines.policy_baselines import (
    build_optimized_network_policy_snapshot,
)
from src.simulation.policy_runner import (
    run_multiple_stochastic_replications,
)

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

N_REPLICATIONS = 200
BASE_SEED = 123

SCENARIOS = [
    "baseline",
    "h3_icu_capacity_reduced",
    "transfer_disabled",
]

ROBUST_CONFIG = RobustArrivalConfig(
    default_rho=0.10,
    rho_by_hospital={"H3": 0.20},
    scenario_multiplier=1.05,
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _clone_data_wrapper(original_wrapper: Any, new_df: pd.DataFrame) -> Any:
    wrapper_cls = type(original_wrapper)
    return wrapper_cls(df=new_df)


def _clone_instance(
    instance: HealthcareInstance,
    *,
    capacities_df: pd.DataFrame | None = None,
    transfer_lanes_df: pd.DataFrame | None = None,
) -> HealthcareInstance:
    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=_clone_data_wrapper(
            instance.capacities,
            instance.capacities.df.copy() if capacities_df is None else capacities_df,
        ),
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=_clone_data_wrapper(
            instance.transfer_lanes,
            instance.transfer_lanes.df.copy() if transfer_lanes_df is None else transfer_lanes_df,
        ),
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=instance.arrivals,
        transitions=instance.transitions,
    )


def extract_kpi_replications(outputs) -> pd.DataFrame:
    if isinstance(outputs, pd.DataFrame):
        return outputs.copy()

    if isinstance(outputs, dict):
        if "kpi_replications" in outputs:
            return outputs["kpi_replications"].copy()
        raise ValueError(
            f"run_multiple_stochastic_replications returned a dict without 'kpi_replications'. "
            f"Keys found: {list(outputs.keys())}"
        )

    raise TypeError(
        f"Unexpected output type from run_multiple_stochastic_replications: {type(outputs)}"
    )


def summarize_outputs(
    scenario: str,
    policy_name: str,
    outputs_df: pd.DataFrame,
) -> dict:
    row = {
        "scenario": scenario,
        "policy_name": policy_name,
        "n_replications": len(outputs_df),
    }

    numeric_cols = [
        "total_unsafe_excess",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
        "total_overflow_excess",
        "total_surge_gap",
    ]

    for col in numeric_cols:
        if col in outputs_df.columns:
            row[f"{col}_mean"] = safe_float(outputs_df[col].mean())
            row[f"{col}_std"] = safe_float(outputs_df[col].std())

    return row


def compare_two_rows(
    nominal_row: pd.Series,
    robust_row: pd.Series,
) -> dict:
    metrics = [
        "total_unsafe_excess_mean",
        "max_utilization_ratio_mean",
        "num_unsafe_rows_mean",
        "total_blocked_arrivals_mean",
        "total_overflow_excess_mean",
        "total_surge_gap_mean",
    ]

    out = {
        "scenario": nominal_row["scenario"],
    }

    for m in metrics:
        if m in nominal_row.index and m in robust_row.index:
            out[f"{m}_nominal"] = safe_float(nominal_row[m])
            out[f"{m}_robust"] = safe_float(robust_row[m])
            out[f"{m}_delta_robust_minus_nominal"] = (
                safe_float(robust_row[m]) - safe_float(nominal_row[m])
            )

    return out


# ============================================================
# SCENARIO MODIFIERS
# ============================================================

def apply_scenario(instance: HealthcareInstance, scenario_name: str) -> HealthcareInstance:
    if scenario_name == "baseline":
        return instance

    if scenario_name == "h3_icu_capacity_reduced":
        cap = instance.capacities.df.copy()

        if "hospital_id" in cap.columns and "resource" in cap.columns:
            mask = (
                (cap["hospital_id"] == "H3")
                & (cap["resource"].astype(str).str.upper() == "ICU")
            )

            if "capacity" in cap.columns:
                cap["capacity"] = cap["capacity"].astype(float)
                cap.loc[mask, "capacity"] = cap.loc[mask, "capacity"] * 0.75

            elif "base_capacity" in cap.columns:
                cap["base_capacity"] = cap["base_capacity"].astype(float)
                cap.loc[mask, "base_capacity"] = cap.loc[mask, "base_capacity"] * 0.75

        return _clone_instance(instance, capacities_df=cap)

    if scenario_name == "transfer_disabled":
        lanes = instance.transfer_lanes.df.copy()

        cap_col = None
        for c in ["lane_capacity", "capacity", "max_transfer", "transfer_capacity", "allowed"]:
            if c in lanes.columns:
                cap_col = c
                break

        if cap_col is not None:
            lanes[cap_col] = 0.0

        return _clone_instance(instance, transfer_lanes_df=lanes)

    return instance


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_one_scenario(scenario_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance = load_healthcare_instance()
    instance = apply_scenario(instance, scenario_name)

    nominal_snapshot = build_optimized_network_policy_snapshot(instance)

    robust_snapshot = build_robust_optimized_network_policy_snapshot(
        instance,
        robust_config=ROBUST_CONFIG,
    )

    nominal_outputs_raw = run_multiple_stochastic_replications(
        instance=instance,
        policy_snapshot=nominal_snapshot,
        n_replications=N_REPLICATIONS,
        base_seed=BASE_SEED,
    )

    robust_outputs_raw = run_multiple_stochastic_replications(
        instance=instance,
        policy_snapshot=robust_snapshot,
        n_replications=N_REPLICATIONS,
        base_seed=BASE_SEED,
    )

    nominal_outputs = extract_kpi_replications(nominal_outputs_raw)
    robust_outputs = extract_kpi_replications(robust_outputs_raw)

    nominal_summary = summarize_outputs(
        scenario=scenario_name,
        policy_name="optimized_network",
        outputs_df=nominal_outputs,
    )

    robust_summary = summarize_outputs(
        scenario=scenario_name,
        policy_name="robust_optimized_network",
        outputs_df=robust_outputs,
    )

    return (
        pd.DataFrame([nominal_summary, robust_summary]),
        pd.concat(
            [
                nominal_outputs.assign(
                    scenario=scenario_name,
                    policy_name="optimized_network",
                ),
                robust_outputs.assign(
                    scenario=scenario_name,
                    policy_name="robust_optimized_network",
                ),
            ],
            ignore_index=True,
        ),
    )


def main():
    all_summary = []
    all_replications = []

    print("\nROBUST VS NOMINAL COMPARISON")
    print("=" * 60)

    for scenario in SCENARIOS:
        print(f"Running scenario: {scenario}")

        summary_df, repl_df = run_one_scenario(
            scenario_name=scenario,
        )

        all_summary.append(summary_df)
        all_replications.append(repl_df)

    summary = pd.concat(all_summary, ignore_index=True)
    replications = pd.concat(all_replications, ignore_index=True)

    delta_rows = []

    for scenario in SCENARIOS:
        sdf = summary[summary["scenario"] == scenario].copy()

        nominal = sdf[sdf["policy_name"] == "optimized_network"].iloc[0]
        robust = sdf[sdf["policy_name"] == "robust_optimized_network"].iloc[0]

        delta_rows.append(compare_two_rows(nominal, robust))

    delta_df = pd.DataFrame(delta_rows)

    summary_path = TABLES_DIR / "robust_vs_nominal_summary.csv"
    repl_path = TABLES_DIR / "robust_vs_nominal_replications.csv"
    delta_path = TABLES_DIR / "robust_vs_nominal_delta.csv"

    summary.to_csv(summary_path, index=False)
    replications.to_csv(repl_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    print("\nSaved tables:")
    print(summary_path)
    print(repl_path)
    print(delta_path)

    print("\n=== Summary ===")
    print(summary)

    print("\n=== Robust minus Nominal Delta ===")
    print(delta_df)

    if "total_unsafe_excess_mean_delta_robust_minus_nominal" in delta_df.columns:
        wins = (
            delta_df["total_unsafe_excess_mean_delta_robust_minus_nominal"] < 0
        ).sum()

        print(
            f"\nRobust policy improved unsafe excess in {wins}/{len(delta_df)} scenarios."
        )


if __name__ == "__main__":
    main()