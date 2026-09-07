from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

from src.baselines.policy_baselines import build_optimized_network_policy_snapshot
from src.config.dataclasses import HealthcareInstance
from src.optimization.robust_uncertainty import (
    RobustArrivalConfig,
    make_robustified_arrivals,
)


def _clone_data_wrapper(original_wrapper: Any, new_df: pd.DataFrame) -> Any:
    wrapper_cls = type(original_wrapper)
    return wrapper_cls(df=new_df)


def _clone_instance_with_arrivals(
    instance: HealthcareInstance,
    arrivals_df: pd.DataFrame,
) -> HealthcareInstance:
    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=instance.capacities,
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=instance.transfer_lanes,
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=_clone_data_wrapper(instance.arrivals, arrivals_df),
        transitions=instance.transitions,
    )


def _safe_copy_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df.copy()


def _attach_robust_metadata_to_metrics(
    metrics_df: pd.DataFrame,
    config: RobustArrivalConfig,
    robust_table: pd.DataFrame,
) -> pd.DataFrame:
    out = _safe_copy_df(metrics_df)

    if out.empty:
        out = pd.DataFrame([{}])

    nominal_total = float(robust_table["nominal_arrival"].sum()) if not robust_table.empty else 0.0
    robust_total = float(robust_table["robust_arrival"].sum()) if not robust_table.empty else 0.0
    delta_total = float(robust_table["robust_delta"].sum()) if not robust_table.empty else 0.0
    mean_rho = float(robust_table["robust_rho"].mean()) if not robust_table.empty else float(config.default_rho)

    out["policy_name"] = "robust_optimized_network"
    out["method_id"] = "arrival_stress_adjusted_nominal_lp"
    out["method_classification"] = "uncertainty_adjusted_nominal_optimization"
    out["formal_robust_optimization"] = False
    out["robust_default_rho"] = float(config.default_rho)
    out["robust_scenario_multiplier"] = float(config.scenario_multiplier)
    out["robust_nominal_arrival_total"] = nominal_total
    out["robust_arrival_total"] = robust_total
    out["robust_arrival_delta_total"] = delta_total
    out["robust_mean_rho"] = mean_rho

    return out


def _attach_policy_name(
    policy_snapshot: dict[str, Any],
    policy_name: str,
) -> dict[str, Any]:
    out = dict(policy_snapshot)

    # Top-level policy name
    out["policy_name"] = policy_name

    # Also patch metrics table if present
    if "model_metrics" in out and isinstance(out["model_metrics"], pd.DataFrame):
        mm = out["model_metrics"].copy()
        if mm.empty:
            mm = pd.DataFrame([{}])
        mm["policy_name"] = policy_name
        out["model_metrics"] = mm

    return out


def _build_robust_summary_tables(
    robust_summary: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        "robust_arrival_summary_overall": _safe_copy_df(robust_summary.get("overall")),
        "robust_arrival_summary_by_hospital": _safe_copy_df(robust_summary.get("by_hospital")),
        "robust_arrival_summary_by_hospital_cohort": _safe_copy_df(
            robust_summary.get("by_hospital_cohort")
        ),
    }


def build_robust_optimized_network_policy_snapshot(
    instance: HealthcareInstance,
    robust_config: RobustArrivalConfig | None = None,
) -> dict[str, Any]:
    """
    Legacy compatibility wrapper for arrival-stress-adjusted optimization.

    This method inflates uncertain arrivals before solving the nominal deterministic
    LP. It is not a formal robust-optimization counterpart with an uncertainty set
    or adversarial subproblem. New code should prefer
    ``build_uncertainty_adjusted_network_policy_snapshot`` from
    ``src.optimization.uncertainty_adjusted_network``. The legacy policy name is
    retained here to preserve stored experiment compatibility.
    """
    if robust_config is None:
        robust_config = RobustArrivalConfig(default_rho=0.10)

    robust_arrivals_df, robust_table, robust_summary = make_robustified_arrivals(
        arrivals_df=instance.arrivals.df,
        config=robust_config,
    )

    robust_instance = _clone_instance_with_arrivals(
        instance=instance,
        arrivals_df=robust_arrivals_df,
    )

    nominal_snapshot = build_optimized_network_policy_snapshot(robust_instance)
    robust_snapshot = _attach_policy_name(nominal_snapshot, "robust_optimized_network")

    # Attach robust metadata to model metrics
    if "model_metrics" in robust_snapshot and isinstance(robust_snapshot["model_metrics"], pd.DataFrame):
        robust_snapshot["model_metrics"] = _attach_robust_metadata_to_metrics(
            robust_snapshot["model_metrics"],
            robust_config,
            robust_table,
        )
    else:
        robust_snapshot["model_metrics"] = _attach_robust_metadata_to_metrics(
            pd.DataFrame(),
            robust_config,
            robust_table,
        )

    # Diagnostics
    robust_snapshot["robust_arrivals_input"] = robust_arrivals_df.copy()
    robust_snapshot["robust_arrival_table"] = robust_table.copy()
    robust_snapshot.update(_build_robust_summary_tables(robust_summary))

    robust_snapshot["method_metadata"] = {
        "method_id": "arrival_stress_adjusted_nominal_lp",
        "method_classification": "uncertainty_adjusted_nominal_optimization",
        "formal_robust_optimization": False,
        "approved_name": "uncertainty-adjusted network optimization",
        "legacy_name": "robust_optimized_network",
    }

    # Keep config for traceability
    if is_dataclass(robust_config):
        robust_snapshot["robust_config"] = asdict(robust_config)
    else:
        robust_snapshot["robust_config"] = {
            "default_rho": float(getattr(robust_config, "default_rho", 0.10)),
            "scenario_multiplier": float(getattr(robust_config, "scenario_multiplier", 1.0)),
        }

    return robust_snapshot


def build_robust_policy_from_named_profile(
    instance: HealthcareInstance,
    profile_name: str = "moderate",
) -> dict[str, Any]:
    """
    Convenience wrapper for named stress profiles.
    """
    from src.optimization.robust_uncertainty import build_stress_test_configs

    profiles = build_stress_test_configs()
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown robust profile '{profile_name}'. "
            f"Available profiles: {sorted(profiles.keys())}"
        )

    return build_robust_optimized_network_policy_snapshot(
        instance=instance,
        robust_config=profiles[profile_name],
    )


def summarize_robust_vs_nominal_policy_snapshot(
    nominal_snapshot: dict[str, Any],
    robust_snapshot: dict[str, Any],
) -> pd.DataFrame:
    """
    Lightweight comparison table between nominal optimized_network and robust_optimized_network
    using model_metrics when available.
    """
    nominal_metrics = nominal_snapshot.get("model_metrics", pd.DataFrame()).copy()
    robust_metrics = robust_snapshot.get("model_metrics", pd.DataFrame()).copy()

    if nominal_metrics.empty and robust_metrics.empty:
        return pd.DataFrame()

    if nominal_metrics.empty:
        nominal_metrics = pd.DataFrame([{"policy_name": "optimized_network"}])

    if robust_metrics.empty:
        robust_metrics = pd.DataFrame([{"policy_name": "robust_optimized_network"}])

    nominal_metrics["comparison_side"] = "nominal"
    robust_metrics["comparison_side"] = "robust"

    stacked = pd.concat([nominal_metrics, robust_metrics], ignore_index=True)

    preferred_cols = [
        "comparison_side",
        "policy_name",
        "objective_value",
        "total_unsafe_excess",
        "total_overflow_excess",
        "total_surge_gap",
        "robust_nominal_arrival_total",
        "robust_arrival_total",
        "robust_arrival_delta_total",
        "robust_mean_rho",
    ]
    cols = [c for c in preferred_cols if c in stacked.columns]
    if cols:
        return stacked[cols].copy()

    return stacked.copy()