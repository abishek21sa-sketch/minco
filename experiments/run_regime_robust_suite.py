from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.policy_baselines import (
    build_local_only_policy_snapshot,
    build_myopic_milp_policy_snapshot,
    build_no_control_policy_snapshot,
    build_no_transfer_policy_snapshot,
    build_optimized_network_policy_snapshot,
)
from src.config.dataclasses import (
    CapacitiesData,
    HealthcareInstance,
    TransferLanesData,
)
from src.config.loader import load_healthcare_instance
from src.optimization.robust_minco import (
    build_robust_optimized_network_policy_snapshot,
)
from src.optimization.regime_robust_minco import (
    build_regime_robust_policy_snapshot,
)
from src.simulation.policy_runner import run_multiple_stochastic_replications


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regime robust scenario suite."
    )

    parser.add_argument(
        "--replications",
        type=int,
        default=200,
        help="Number of simulation replications per scenario/policy.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Lookahead horizon for regime robust policy.",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def _clone_instance(
    instance: HealthcareInstance,
    *,
    capacities_df: pd.DataFrame | None = None,
    transfer_lanes_df: pd.DataFrame | None = None,
) -> HealthcareInstance:
    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=CapacitiesData(
            df=instance.capacities.df.copy() if capacities_df is None else capacities_df
        ),
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=TransferLanesData(
            df=instance.transfer_lanes.df.copy() if transfer_lanes_df is None else transfer_lanes_df
        ),
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=instance.arrivals,
        transitions=instance.transitions,
    )


def extract_kpi_replications(outputs: Any) -> pd.DataFrame:
    if isinstance(outputs, pd.DataFrame):
        return outputs.copy()

    if isinstance(outputs, dict):
        if "kpi_replications" in outputs:
            return outputs["kpi_replications"].copy()
        raise ValueError(
            f"run_multiple_stochastic_replications returned dict without "
            f"'kpi_replications'. Keys found: {list(outputs.keys())}"
        )

    raise TypeError(
        f"Unexpected output type from run_multiple_stochastic_replications: {type(outputs)}"
    )


# ============================================================
# Scenario handling
# ============================================================

def apply_scenario(instance: HealthcareInstance, scenario_name: str) -> HealthcareInstance:
    cap = instance.capacities.df.copy()
    lanes = instance.transfer_lanes.df.copy()

    capacity_col = "base_capacity" if "base_capacity" in cap.columns else "capacity"
    hospital_col = "hospital_id" if "hospital_id" in cap.columns else "hospital"
    resource_col = "resource" if "resource" in cap.columns else "unit"

    cap[capacity_col] = pd.to_numeric(cap[capacity_col], errors="coerce").astype(float)

    if scenario_name == "baseline":
        return instance

    if scenario_name == "h3_icu_capacity_reduced":
        mask = (
            (cap[hospital_col] == "H3")
            & (cap[resource_col].astype(str).str.upper() == "ICU")
        )
        if mask.any():
            cap.loc[mask, capacity_col] *= 0.75

        return _clone_instance(instance, capacities_df=cap)

    if scenario_name == "transfer_disabled":
        lane_col = None
        for c in ["allowed", "capacity", "max_transfer", "transfer_capacity", "lane_capacity"]:
            if c in lanes.columns:
                lane_col = c
                break

        if lane_col is None:
            lanes["allowed"] = 0
        else:
            lanes[lane_col] = 0

        return _clone_instance(instance, transfer_lanes_df=lanes)

    if scenario_name == "network_stress":
        mask = cap[resource_col].astype(str).str.upper().isin(["ICU", "WARD"])
        cap.loc[mask, capacity_col] *= 0.90
        return _clone_instance(instance, capacities_df=cap)

    if scenario_name == "regional_crisis":
        mask = cap[resource_col].astype(str).str.upper() == "ICU"
        cap.loc[mask, capacity_col] *= 0.80
        return _clone_instance(instance, capacities_df=cap)

    raise ValueError(f"Unknown scenario '{scenario_name}'")


# ============================================================
# Policy builders
# ============================================================

def build_policy_dict(
    instance: HealthcareInstance,
    regime_for_scenario: str,
    lookahead_horizon: int,
) -> dict[str, dict]:
    policies: dict[str, dict] = {}

    policies["no_control"] = build_no_control_policy_snapshot(instance)
    policies["local_only"] = build_local_only_policy_snapshot(instance)
    policies["no_transfer"] = build_no_transfer_policy_snapshot(instance)
    policies["myopic_milp"] = build_myopic_milp_policy_snapshot(instance)
    policies["optimized_network"] = build_optimized_network_policy_snapshot(instance)

    policies["robust_optimized_network"] = (
        build_robust_optimized_network_policy_snapshot(instance)
    )

    policies["regime_robust_optimized_network"] = (
        build_regime_robust_policy_snapshot(
            instance=instance,
            current_regime=regime_for_scenario,
            lookahead_horizon=lookahead_horizon,
            use_expected_weights=True,
        )
    )

    return policies


# ============================================================
# Scenario -> regime mapping
# ============================================================

def scenario_regime_map() -> dict[str, str]:
    return {
        "baseline": "normal",
        "h3_icu_capacity_reduced": "surge",
        "transfer_disabled": "surge",
        "network_stress": "surge",
        "regional_crisis": "crisis",
    }


# ============================================================
# Summaries
# ============================================================

def summarize_replications(
    rep_df: pd.DataFrame,
    scenario: str,
    policy_name: str,
) -> pd.DataFrame:
    if rep_df.empty:
        return pd.DataFrame()

    metric_cols = [
        c for c in rep_df.columns
        if c not in ["replication", "scenario", "policy_name"]
    ]

    row = {
        "scenario": scenario,
        "policy_name": policy_name,
        "n_replications": len(rep_df),
    }

    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(rep_df[col]):
            row[f"{col}_mean"] = float(rep_df[col].mean())
            row[f"{col}_std"] = float(rep_df[col].std(ddof=1))

    return pd.DataFrame([row])


# ============================================================
# One scenario
# ============================================================

def run_one_scenario(
    base_instance: HealthcareInstance,
    scenario_name: str,
    replications: int,
    seed: int,
    lookahead_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    instance = apply_scenario(base_instance, scenario_name)

    regime_lookup = scenario_regime_map()
    current_regime = regime_lookup[scenario_name]

    policy_dict = build_policy_dict(
        instance=instance,
        regime_for_scenario=current_regime,
        lookahead_horizon=lookahead_horizon,
    )

    all_summary = []
    all_replications = []

    for i, (policy_name, snapshot) in enumerate(policy_dict.items()):
        outputs = run_multiple_stochastic_replications(
            instance=instance,
            policy_snapshot=snapshot,
            n_replications=replications,
            base_seed=seed + i * 1000,
        )

        rep_df = extract_kpi_replications(outputs)
        rep_df["scenario"] = scenario_name
        rep_df["policy_name"] = policy_name

        summary_df = summarize_replications(
            rep_df=rep_df,
            scenario=scenario_name,
            policy_name=policy_name,
        )

        all_summary.append(summary_df)
        all_replications.append(rep_df)

    summary_out = pd.concat(all_summary, ignore_index=True)
    repl_out = pd.concat(all_replications, ignore_index=True)

    return summary_out, repl_out


# ============================================================
# Comparison tables
# ============================================================

def build_regime_vs_nominal_delta(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    scenarios = summary_df["scenario"].dropna().unique().tolist()

    for scenario in scenarios:
        sdf = summary_df[summary_df["scenario"] == scenario].copy()

        nominal = sdf[sdf["policy_name"] == "optimized_network"]
        robust = sdf[sdf["policy_name"] == "robust_optimized_network"]
        regime = sdf[sdf["policy_name"] == "regime_robust_optimized_network"]

        if nominal.empty or robust.empty or regime.empty:
            continue

        n = nominal.iloc[0]
        r = robust.iloc[0]
        g = regime.iloc[0]

        rows.append(
            {
                "scenario": scenario,
                "unsafe_nominal": n.get("total_unsafe_excess_mean", np.nan),
                "unsafe_robust": r.get("total_unsafe_excess_mean", np.nan),
                "unsafe_regime": g.get("total_unsafe_excess_mean", np.nan),
                "delta_regime_minus_nominal": (
                    g.get("total_unsafe_excess_mean", np.nan)
                    - n.get("total_unsafe_excess_mean", np.nan)
                ),
                "delta_regime_minus_robust": (
                    g.get("total_unsafe_excess_mean", np.nan)
                    - r.get("total_unsafe_excess_mean", np.nan)
                ),
                "blocked_regime": g.get("total_blocked_arrivals_mean", np.nan),
                "util_regime": g.get("max_utilization_ratio_mean", np.nan),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    print("\nREGIME ROBUST SCENARIO SUITE")
    print("=" * 70)

    base_instance = load_healthcare_instance()

    scenarios = [
        "baseline",
        "h3_icu_capacity_reduced",
        "transfer_disabled",
        "network_stress",
        "regional_crisis",
    ]

    all_summary = []
    all_replications = []

    for idx, scenario in enumerate(scenarios):
        print(f"Running scenario: {scenario}")

        summary_df, repl_df = run_one_scenario(
            base_instance=base_instance,
            scenario_name=scenario,
            replications=args.replications,
            seed=args.seed + idx * 5000,
            lookahead_horizon=args.horizon,
        )

        all_summary.append(summary_df)
        all_replications.append(repl_df)

    summary_final = pd.concat(all_summary, ignore_index=True)
    repl_final = pd.concat(all_replications, ignore_index=True)

    delta_df = build_regime_vs_nominal_delta(summary_final)

    summary_path = TABLES_DIR / "regime_suite_summary.csv"
    repl_path = TABLES_DIR / "regime_suite_replications.csv"
    delta_path = TABLES_DIR / "regime_suite_delta.csv"

    summary_final.to_csv(summary_path, index=False)
    repl_final.to_csv(repl_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    print("\nSaved tables:")
    print(summary_path)
    print(repl_path)
    print(delta_path)

    print("\n=== Headline Results ===")
    if not delta_df.empty:
        print(delta_df.to_string(index=False))
        improved = (delta_df["delta_regime_minus_nominal"] < 0).sum()
        print(
            f"\nRegime robust improved unsafe excess vs nominal in "
            f"{improved}/{len(delta_df)} scenarios."
        )
    else:
        print("No comparison rows generated.")


if __name__ == "__main__":
    main()