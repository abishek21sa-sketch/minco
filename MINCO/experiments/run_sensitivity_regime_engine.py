from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.policy_baselines import (
    build_optimized_network_policy_snapshot,
)
from src.config.dataclasses import HealthcareInstance
from src.config.loader import load_healthcare_instance
from src.optimization.regime_robust_minco import (
    build_regime_robust_policy_snapshot,
)
from src.optimization.robust_minco import (
    build_robust_optimized_network_policy_snapshot,
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
        description="Run regime-aware policy sensitivity engine."
    )

    parser.add_argument(
        "--replications",
        type=int,
        default=100,
        help="Number of stochastic replications per design point and policy.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed.",
    )

    return parser.parse_args()


# ============================================================
# Generic helpers
# ============================================================

def _clone_data_wrapper(original_wrapper: Any, new_df: pd.DataFrame) -> Any:
    wrapper_cls = type(original_wrapper)
    return wrapper_cls(df=new_df)


def clone_instance(
    instance: HealthcareInstance,
    *,
    capacities_df: pd.DataFrame | None = None,
    transfer_lanes_df: pd.DataFrame | None = None,
    arrivals_df: pd.DataFrame | None = None,
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
        arrivals=_clone_data_wrapper(
            instance.arrivals,
            instance.arrivals.df.copy() if arrivals_df is None else arrivals_df,
        ),
        transitions=instance.transitions,
    )


def extract_kpi_replications(outputs: Any) -> pd.DataFrame:
    if isinstance(outputs, pd.DataFrame):
        return outputs.copy()

    if isinstance(outputs, dict):
        if "kpi_replications" in outputs:
            return outputs["kpi_replications"].copy()
        raise ValueError(
            f"Expected 'kpi_replications' in outputs. Keys found: {list(outputs.keys())}"
        )

    raise TypeError(
        f"Unexpected output type from run_multiple_stochastic_replications: {type(outputs)}"
    )


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# Parameter perturbations
# ============================================================

def scale_icu_capacity(instance: HealthcareInstance, scale: float) -> HealthcareInstance:
    cap = instance.capacities.df.copy()

    hospital_col = first_existing(cap, ["hospital_id", "hospital"])
    resource_col = first_existing(cap, ["resource", "unit"])
    capacity_col = first_existing(cap, ["base_capacity", "capacity"])

    if resource_col is None or capacity_col is None:
        return instance

    cap[capacity_col] = pd.to_numeric(cap[capacity_col], errors="coerce").astype(float)

    mask = cap[resource_col].astype(str).str.upper() == "ICU"
    cap.loc[mask, capacity_col] = cap.loc[mask, capacity_col] * float(scale)

    return clone_instance(instance, capacities_df=cap)


def scale_transfer_capacity(instance: HealthcareInstance, scale: float) -> HealthcareInstance:
    lanes = instance.transfer_lanes.df.copy()

    lane_col = first_existing(
        lanes,
        ["capacity", "lane_capacity", "max_transfer", "transfer_capacity"],
    )

    if lane_col is None:
        if "allowed" in lanes.columns:
            if scale <= 0:
                lanes["allowed"] = 0
            return clone_instance(instance, transfer_lanes_df=lanes)
        return instance

    lanes[lane_col] = pd.to_numeric(lanes[lane_col], errors="coerce").astype(float)
    lanes[lane_col] = lanes[lane_col] * float(scale)

    return clone_instance(instance, transfer_lanes_df=lanes)


def scale_arrivals(instance: HealthcareInstance, scale: float) -> HealthcareInstance:
    arr = instance.arrivals.df.copy()

    arrival_col = first_existing(
        arr,
        ["arrival_rate", "arrivals", "lambda", "arrival_volume", "mean_arrivals"],
    )
    if arrival_col is None:
        return instance

    arr[arrival_col] = pd.to_numeric(arr[arrival_col], errors="coerce").astype(float)
    arr[arrival_col] = arr[arrival_col] * float(scale)

    return clone_instance(instance, arrivals_df=arr)


def apply_design_point(
    base_instance: HealthcareInstance,
    *,
    icu_capacity_scale: float,
    transfer_capacity_scale: float,
    demand_scale: float,
) -> HealthcareInstance:
    inst = base_instance
    inst = scale_icu_capacity(inst, icu_capacity_scale)
    inst = scale_transfer_capacity(inst, transfer_capacity_scale)
    inst = scale_arrivals(inst, demand_scale)
    return inst


# ============================================================
# Policy builders
# ============================================================

def build_policy_dict(
    instance: HealthcareInstance,
    *,
    regime: str,
    lookahead_horizon: int,
    uncertainty_profile: str,
) -> dict[str, dict]:
    # profile -> wrappers
    if uncertainty_profile == "mild":
        robust_snapshot = build_robust_optimized_network_policy_snapshot(
            instance,
        )
        regime_snapshot = build_regime_robust_policy_snapshot(
            instance=instance,
            current_regime=regime,
            lookahead_horizon=lookahead_horizon,
            use_expected_weights=True,
            scenario_multiplier=1.00,
            rho_by_hospital={},
            rho_by_hospital_cohort={},
        )

    elif uncertainty_profile == "moderate":
        robust_snapshot = build_robust_optimized_network_policy_snapshot(
            instance,
        )
        regime_snapshot = build_regime_robust_policy_snapshot(
            instance=instance,
            current_regime=regime,
            lookahead_horizon=lookahead_horizon,
            use_expected_weights=True,
            scenario_multiplier=1.03,
            rho_by_hospital={"H3": 0.20},
            rho_by_hospital_cohort={},
        )

    elif uncertainty_profile == "severe":
        robust_snapshot = build_robust_optimized_network_policy_snapshot(
            instance,
        )
        regime_snapshot = build_regime_robust_policy_snapshot(
            instance=instance,
            current_regime=regime,
            lookahead_horizon=lookahead_horizon,
            use_expected_weights=True,
            scenario_multiplier=1.08,
            rho_by_hospital={"H2": 0.15, "H3": 0.25},
            rho_by_hospital_cohort={"H3|c3": 0.30},
        )

    else:
        raise ValueError(f"Unknown uncertainty_profile '{uncertainty_profile}'")

    return {
        "optimized_network": build_optimized_network_policy_snapshot(instance),
        "robust_optimized_network": robust_snapshot,
        "regime_robust_optimized_network": regime_snapshot,
    }


# ============================================================
# Experimental design
# ============================================================

def build_design_table() -> pd.DataFrame:
    rows = []

    # one-factor-at-a-time around a reference point
    base = {
        "design_name": "base",
        "icu_capacity_scale": 1.00,
        "transfer_capacity_scale": 1.00,
        "demand_scale": 1.00,
        "regime": "surge",
        "lookahead_horizon": 7,
        "uncertainty_profile": "moderate",
    }
    rows.append(base)

    rows.extend(
        [
            {
                "design_name": "icu_minus_20",
                "icu_capacity_scale": 0.80,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "icu_plus_20",
                "icu_capacity_scale": 1.20,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "transfer_minus_50",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 0.50,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "transfer_plus_50",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.50,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "demand_minus_20",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 0.80,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "demand_plus_20",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.20,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "normal_regime",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "normal",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "crisis_regime",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "crisis",
                "lookahead_horizon": 7,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "short_horizon",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 3,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "long_horizon",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 10,
                "uncertainty_profile": "moderate",
            },
            {
                "design_name": "mild_uncertainty",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "mild",
            },
            {
                "design_name": "severe_uncertainty",
                "icu_capacity_scale": 1.00,
                "transfer_capacity_scale": 1.00,
                "demand_scale": 1.00,
                "regime": "surge",
                "lookahead_horizon": 7,
                "uncertainty_profile": "severe",
            },
        ]
    )

    return pd.DataFrame(rows)


# ============================================================
# Summaries
# ============================================================

def summarize_replications(
    rep_df: pd.DataFrame,
    *,
    design_name: str,
    policy_name: str,
    regime: str,
    lookahead_horizon: int,
    uncertainty_profile: str,
    icu_capacity_scale: float,
    transfer_capacity_scale: float,
    demand_scale: float,
) -> pd.DataFrame:
    if rep_df.empty:
        return pd.DataFrame()

    row = {
        "design_name": design_name,
        "policy_name": policy_name,
        "regime": regime,
        "lookahead_horizon": lookahead_horizon,
        "uncertainty_profile": uncertainty_profile,
        "icu_capacity_scale": icu_capacity_scale,
        "transfer_capacity_scale": transfer_capacity_scale,
        "demand_scale": demand_scale,
        "n_replications": len(rep_df),
    }

    metric_cols = [
        "total_unsafe_excess",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
        "total_overflow_excess",
        "total_surge_gap",
    ]
    metric_cols = [c for c in metric_cols if c in rep_df.columns]

    for col in metric_cols:
        row[f"{col}_mean"] = float(rep_df[col].mean())
        row[f"{col}_std"] = float(rep_df[col].std(ddof=1)) if len(rep_df) > 1 else 0.0

    return pd.DataFrame([row])


def build_tornado_input(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    base_rows = summary_df[
        (summary_df["design_name"] == "base")
        & (summary_df["policy_name"] == "regime_robust_optimized_network")
    ].copy()

    if base_rows.empty:
        return pd.DataFrame()

    base = base_rows.iloc[0]
    base_unsafe = float(base.get("total_unsafe_excess_mean", np.nan))
    base_blocked = float(base.get("total_blocked_arrivals_mean", np.nan))

    rows = []
    for _, row in summary_df.iterrows():
        if row["policy_name"] != "regime_robust_optimized_network":
            continue

        rows.append(
            {
                "design_name": row["design_name"],
                "unsafe_excess_change_vs_base": (
                    float(row.get("total_unsafe_excess_mean", np.nan)) - base_unsafe
                ),
                "blocked_arrivals_change_vs_base": (
                    float(row.get("total_blocked_arrivals_mean", np.nan)) - base_blocked
                ),
                "regime": row["regime"],
                "lookahead_horizon": row["lookahead_horizon"],
                "uncertainty_profile": row["uncertainty_profile"],
                "icu_capacity_scale": row["icu_capacity_scale"],
                "transfer_capacity_scale": row["transfer_capacity_scale"],
                "demand_scale": row["demand_scale"],
            }
        )

    return pd.DataFrame(rows).sort_values("design_name").reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    print("\nREGIME SENSITIVITY ENGINE")
    print("=" * 70)

    base_instance = load_healthcare_instance()
    design_df = build_design_table()

    all_summary = []
    all_replications = []

    for design_idx, design in design_df.iterrows():
        design_name = str(design["design_name"])
        print(f"Running design: {design_name}")

        instance = apply_design_point(
            base_instance,
            icu_capacity_scale=float(design["icu_capacity_scale"]),
            transfer_capacity_scale=float(design["transfer_capacity_scale"]),
            demand_scale=float(design["demand_scale"]),
        )

        policy_dict = build_policy_dict(
            instance=instance,
            regime=str(design["regime"]),
            lookahead_horizon=int(design["lookahead_horizon"]),
            uncertainty_profile=str(design["uncertainty_profile"]),
        )

        for offset, (policy_name, snapshot) in enumerate(policy_dict.items()):
            outputs = run_multiple_stochastic_replications(
                instance=instance,
                policy_snapshot=snapshot,
                n_replications=args.replications,
                base_seed=args.seed + 10000 * design_idx + 1000 * offset,
            )

            rep_df = extract_kpi_replications(outputs)
            rep_df["design_name"] = design_name
            rep_df["policy_name"] = policy_name
            rep_df["regime"] = str(design["regime"])
            rep_df["lookahead_horizon"] = int(design["lookahead_horizon"])
            rep_df["uncertainty_profile"] = str(design["uncertainty_profile"])
            rep_df["icu_capacity_scale"] = float(design["icu_capacity_scale"])
            rep_df["transfer_capacity_scale"] = float(design["transfer_capacity_scale"])
            rep_df["demand_scale"] = float(design["demand_scale"])

            summary_row = summarize_replications(
                rep_df,
                design_name=design_name,
                policy_name=policy_name,
                regime=str(design["regime"]),
                lookahead_horizon=int(design["lookahead_horizon"]),
                uncertainty_profile=str(design["uncertainty_profile"]),
                icu_capacity_scale=float(design["icu_capacity_scale"]),
                transfer_capacity_scale=float(design["transfer_capacity_scale"]),
                demand_scale=float(design["demand_scale"]),
            )

            all_replications.append(rep_df)
            all_summary.append(summary_row)

    summary_final = pd.concat(all_summary, ignore_index=True)
    replications_final = pd.concat(all_replications, ignore_index=True)
    tornado_df = build_tornado_input(summary_final)

    summary_path = TABLES_DIR / "regime_sensitivity_summary.csv"
    replications_path = TABLES_DIR / "regime_sensitivity_replications.csv"
    tornado_path = TABLES_DIR / "regime_sensitivity_tornado_input.csv"

    summary_final.to_csv(summary_path, index=False)
    replications_final.to_csv(replications_path, index=False)
    tornado_df.to_csv(tornado_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(replications_path)
    print(tornado_path)

    print("\n=== Base design summary ===")
    print(
        summary_final[summary_final["design_name"] == "base"].to_string(index=False)
    )

    print("\n=== Tornado input preview ===")
    print(tornado_df.to_string(index=False))


if __name__ == "__main__":
    main()