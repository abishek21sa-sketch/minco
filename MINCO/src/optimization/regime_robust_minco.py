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
from src.regime.regime_parameters import (
    expected_weighted_arrival_multiplier,
    expected_weighted_uncertainty_width,
    load_default_regime_parameters,
)
from src.regime.regime_process import (
    RegimeProcess,
    build_default_regime_process,
    build_most_likely_regime_path,
    expected_regime_weights,
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


def _get_common_time_col(df: pd.DataFrame) -> str | None:
    for c in ["day", "time", "period", "t"]:
        if c in df.columns:
            return c
    return None


def _get_arrival_col(df: pd.DataFrame) -> str:
    for c in ["arrival_rate", "arrivals", "lambda", "arrival_volume", "mean_arrivals"]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find arrival column in arrivals df. Columns found: {list(df.columns)}"
    )


def _build_step_multiplier_table(
    arrivals_df: pd.DataFrame,
    current_regime: str,
    lookahead_horizon: int,
    regime_process: RegimeProcess | None = None,
    regime_parameters: dict[str, Any] | None = None,
    use_expected_weights: bool = True,
) -> pd.DataFrame:
    if arrivals_df.empty:
        return pd.DataFrame()

    if regime_process is None:
        regime_process = build_default_regime_process()

    if regime_parameters is None:
        regime_parameters = load_default_regime_parameters()

    time_col = _get_common_time_col(arrivals_df)
    if time_col is None:
        # no explicit time column; apply step-0 expected multiplier to all rows
        if use_expected_weights:
            weights0 = expected_regime_weights(
                current_regime=current_regime,
                horizon=1,
                regime_process=regime_process,
            )[0]
            arrival_mult = expected_weighted_arrival_multiplier(weights0, regime_parameters)
            unc_width = expected_weighted_uncertainty_width(weights0, regime_parameters)
        else:
            path = build_most_likely_regime_path(
                horizon=1,
                start_regime=current_regime,
                regime_process=regime_process,
            )
            g0 = path[0]
            arrival_mult = regime_parameters[g0].arrival_multiplier
            unc_width = regime_parameters[g0].arrival_uncertainty_width

        return pd.DataFrame(
            {
                "_step_proxy": [0],
                "regime_arrival_multiplier": [float(arrival_mult)],
                "regime_uncertainty_width": [float(unc_width)],
            }
        )

    unique_steps = sorted(arrivals_df[time_col].dropna().unique().tolist())
    if lookahead_horizon <= 0:
        lookahead_horizon = max(len(unique_steps), 1)

    rows = []

    if use_expected_weights:
        weights_by_step = expected_regime_weights(
            current_regime=current_regime,
            horizon=lookahead_horizon,
            regime_process=regime_process,
        )

        for i, step_val in enumerate(unique_steps):
            weights = weights_by_step.get(i, weights_by_step.get(lookahead_horizon - 1, {}))
            if not weights:
                weights = {current_regime: 1.0}

            arrival_mult = expected_weighted_arrival_multiplier(weights, regime_parameters)
            unc_width = expected_weighted_uncertainty_width(weights, regime_parameters)

            rows.append(
                {
                    time_col: step_val,
                    "regime_arrival_multiplier": float(arrival_mult),
                    "regime_uncertainty_width": float(unc_width),
                }
            )
    else:
        path = build_most_likely_regime_path(
            horizon=max(lookahead_horizon, len(unique_steps)),
            start_regime=current_regime,
            regime_process=regime_process,
        )

        for i, step_val in enumerate(unique_steps):
            regime = path[min(i, len(path) - 1)]
            params = regime_parameters[regime]

            rows.append(
                {
                    time_col: step_val,
                    "regime_arrival_multiplier": float(params.arrival_multiplier),
                    "regime_uncertainty_width": float(params.arrival_uncertainty_width),
                }
            )

    return pd.DataFrame(rows)


def build_regime_adjusted_arrivals(
    arrivals_df: pd.DataFrame,
    current_regime: str,
    lookahead_horizon: int = 7,
    regime_process: RegimeProcess | None = None,
    regime_parameters: dict[str, Any] | None = None,
    use_expected_weights: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies regime-conditioned expected arrival multipliers to the arrivals table.
    Returns:
        adjusted_arrivals_df,
        step_multiplier_table
    """
    if arrivals_df.empty:
        return arrivals_df.copy(), pd.DataFrame()

    if regime_process is None:
        regime_process = build_default_regime_process()

    if regime_parameters is None:
        regime_parameters = load_default_regime_parameters()

    df = arrivals_df.copy()
    arrival_col = _get_arrival_col(df)
    df[arrival_col] = pd.to_numeric(df[arrival_col], errors="coerce").fillna(0.0)

    time_col = _get_common_time_col(df)
    step_multiplier_table = _build_step_multiplier_table(
        arrivals_df=df,
        current_regime=current_regime,
        lookahead_horizon=lookahead_horizon,
        regime_process=regime_process,
        regime_parameters=regime_parameters,
        use_expected_weights=use_expected_weights,
    )

    if time_col is None:
        mult = float(step_multiplier_table["regime_arrival_multiplier"].iloc[0])
        df["nominal_arrival_pre_regime"] = df[arrival_col].astype(float)
        df[arrival_col] = df[arrival_col] * mult
        df["regime_arrival_multiplier"] = mult
        return df, step_multiplier_table

    df = df.merge(step_multiplier_table, on=time_col, how="left", validate="many_to_one")

    if df["regime_arrival_multiplier"].isnull().any():
        raise ValueError("Failed to merge regime arrival multipliers onto arrivals df.")

    df["nominal_arrival_pre_regime"] = df[arrival_col].astype(float)
    df[arrival_col] = df[arrival_col] * df["regime_arrival_multiplier"]

    return df, step_multiplier_table


def build_regime_robust_arrivals(
    arrivals_df: pd.DataFrame,
    current_regime: str,
    lookahead_horizon: int = 7,
    regime_process: RegimeProcess | None = None,
    regime_parameters: dict[str, Any] | None = None,
    use_expected_weights: bool = True,
    rho_by_hospital: dict[str, float] | None = None,
    rho_by_hospital_cohort: dict[str, float] | None = None,
    scenario_multiplier: float = 1.0,
    min_arrival_floor: float = 0.0,
    max_arrival_cap: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Regime-aware robustified arrivals:
    1. regime-adjusted nominal arrivals
    2. robustification using regime-dependent uncertainty width
    """
    if regime_process is None:
        regime_process = build_default_regime_process()

    if regime_parameters is None:
        regime_parameters = load_default_regime_parameters()

    regime_adjusted_df, step_multiplier_table = build_regime_adjusted_arrivals(
        arrivals_df=arrivals_df,
        current_regime=current_regime,
        lookahead_horizon=lookahead_horizon,
        regime_process=regime_process,
        regime_parameters=regime_parameters,
        use_expected_weights=use_expected_weights,
    )

    if step_multiplier_table.empty:
        default_rho = regime_parameters[current_regime].arrival_uncertainty_width
    else:
        default_rho = float(step_multiplier_table["regime_uncertainty_width"].mean())

    robust_config = RobustArrivalConfig(
        default_rho=float(default_rho),
        rho_by_hospital={} if rho_by_hospital is None else rho_by_hospital,
        rho_by_hospital_cohort={} if rho_by_hospital_cohort is None else rho_by_hospital_cohort,
        scenario_multiplier=float(scenario_multiplier),
        min_arrival_floor=float(min_arrival_floor),
        max_arrival_cap=max_arrival_cap,
    )

    robust_arrivals_df, robust_table, robust_summary = make_robustified_arrivals(
        arrivals_df=regime_adjusted_df,
        config=robust_config,
    )

    return robust_arrivals_df, robust_table, robust_summary, step_multiplier_table


def _attach_policy_name(
    policy_snapshot: dict[str, Any],
    policy_name: str,
) -> dict[str, Any]:
    out = dict(policy_snapshot)
    out["policy_name"] = policy_name

    if "model_metrics" in out and isinstance(out["model_metrics"], pd.DataFrame):
        mm = out["model_metrics"].copy()
        if mm.empty:
            mm = pd.DataFrame([{}])
        mm["policy_name"] = policy_name
        out["model_metrics"] = mm

    return out


def _attach_regime_metadata(
    policy_snapshot: dict[str, Any],
    *,
    current_regime: str,
    lookahead_horizon: int,
    use_expected_weights: bool,
    step_multiplier_table: pd.DataFrame,
    robust_arrival_table: pd.DataFrame,
) -> dict[str, Any]:
    out = dict(policy_snapshot)

    if "model_metrics" in out and isinstance(out["model_metrics"], pd.DataFrame):
        mm = out["model_metrics"].copy()
    else:
        mm = pd.DataFrame([{}])

    robust_nominal_total = (
        float(robust_arrival_table["nominal_arrival"].sum())
        if not robust_arrival_table.empty
        else 0.0
    )
    robust_total = (
        float(robust_arrival_table["robust_arrival"].sum())
        if not robust_arrival_table.empty
        else 0.0
    )
    robust_delta_total = (
        float(robust_arrival_table["robust_delta"].sum())
        if not robust_arrival_table.empty
        else 0.0
    )

    mean_regime_mult = (
        float(step_multiplier_table["regime_arrival_multiplier"].mean())
        if not step_multiplier_table.empty and "regime_arrival_multiplier" in step_multiplier_table.columns
        else 1.0
    )
    mean_unc_width = (
        float(step_multiplier_table["regime_uncertainty_width"].mean())
        if not step_multiplier_table.empty and "regime_uncertainty_width" in step_multiplier_table.columns
        else 0.0
    )

    mm["policy_name"] = "regime_robust_optimized_network"
    mm["current_regime"] = str(current_regime)
    mm["lookahead_horizon"] = int(lookahead_horizon)
    mm["use_expected_regime_weights"] = bool(use_expected_weights)
    mm["mean_regime_arrival_multiplier"] = mean_regime_mult
    mm["mean_regime_uncertainty_width"] = mean_unc_width
    mm["regime_robust_nominal_arrival_total"] = robust_nominal_total
    mm["regime_robust_arrival_total"] = robust_total
    mm["regime_robust_arrival_delta_total"] = robust_delta_total

    out["model_metrics"] = mm
    return out


def build_regime_robust_policy_snapshot(
    instance: HealthcareInstance,
    current_regime: str = "normal",
    lookahead_horizon: int = 7,
    regime_process: RegimeProcess | None = None,
    regime_parameters: dict[str, Any] | None = None,
    use_expected_weights: bool = True,
    rho_by_hospital: dict[str, float] | None = None,
    rho_by_hospital_cohort: dict[str, float] | None = None,
    scenario_multiplier: float = 1.0,
    min_arrival_floor: float = 0.0,
    max_arrival_cap: float | None = None,
) -> dict[str, Any]:
    """
    Regime-aware robust MINCO v1:
    - build regime-conditioned arrivals
    - robustify using regime-dependent uncertainty width
    - clone instance with these arrivals
    - call nominal optimized network builder
    - relabel and attach diagnostics
    """
    if regime_process is None:
        regime_process = build_default_regime_process()

    if regime_parameters is None:
        regime_parameters = load_default_regime_parameters()

    robust_arrivals_df, robust_table, robust_summary, step_multiplier_table = build_regime_robust_arrivals(
        arrivals_df=instance.arrivals.df,
        current_regime=current_regime,
        lookahead_horizon=lookahead_horizon,
        regime_process=regime_process,
        regime_parameters=regime_parameters,
        use_expected_weights=use_expected_weights,
        rho_by_hospital=rho_by_hospital,
        rho_by_hospital_cohort=rho_by_hospital_cohort,
        scenario_multiplier=scenario_multiplier,
        min_arrival_floor=min_arrival_floor,
        max_arrival_cap=max_arrival_cap,
    )

    regime_instance = _clone_instance_with_arrivals(
        instance=instance,
        arrivals_df=robust_arrivals_df,
    )

    nominal_snapshot = build_optimized_network_policy_snapshot(regime_instance)
    regime_snapshot = _attach_policy_name(
        nominal_snapshot,
        "regime_robust_optimized_network",
    )

    regime_snapshot = _attach_regime_metadata(
        regime_snapshot,
        current_regime=current_regime,
        lookahead_horizon=lookahead_horizon,
        use_expected_weights=use_expected_weights,
        step_multiplier_table=step_multiplier_table,
        robust_arrival_table=robust_table,
    )

    # diagnostics
    regime_snapshot["regime_robust_arrivals_input"] = robust_arrivals_df.copy()
    regime_snapshot["regime_robust_arrival_table"] = robust_table.copy()
    regime_snapshot["regime_step_multiplier_table"] = step_multiplier_table.copy()
    regime_snapshot["regime_robust_arrival_summary_overall"] = _safe_copy_df(
        robust_summary.get("overall")
    )
    regime_snapshot["regime_robust_arrival_summary_by_hospital"] = _safe_copy_df(
        robust_summary.get("by_hospital")
    )
    regime_snapshot["regime_robust_arrival_summary_by_hospital_cohort"] = _safe_copy_df(
        robust_summary.get("by_hospital_cohort")
    )

    if is_dataclass(regime_process):
        regime_snapshot["regime_process"] = {
            "regimes": list(regime_process.regimes),
            "transition_matrix": regime_process.transition_matrix.copy(),
        }
    else:
        regime_snapshot["regime_process"] = {"note": "custom regime process"}

    regime_snapshot["current_regime"] = str(current_regime)
    regime_snapshot["lookahead_horizon"] = int(lookahead_horizon)
    regime_snapshot["use_expected_regime_weights"] = bool(use_expected_weights)

    if isinstance(regime_parameters, dict):
        regime_snapshot["regime_parameters"] = {
            k: asdict(v) if is_dataclass(v) else v
            for k, v in regime_parameters.items()
        }
    else:
        regime_snapshot["regime_parameters"] = {"note": "custom regime parameters"}

    return regime_snapshot


def build_regime_robust_policy_from_profile(
    instance: HealthcareInstance,
    current_regime: str = "normal",
    profile_name: str = "default",
    lookahead_horizon: int = 7,
    use_expected_weights: bool = True,
) -> dict[str, Any]:
    """
    Simple convenience interface for common regime-aware profiles.
    """
    profile_map = {
        "default": {
            "rho_by_hospital": {},
            "rho_by_hospital_cohort": {},
            "scenario_multiplier": 1.0,
        },
        "h3_guarded": {
            "rho_by_hospital": {"H3": 0.20},
            "rho_by_hospital_cohort": {},
            "scenario_multiplier": 1.0,
        },
        "icu_heavy": {
            "rho_by_hospital": {},
            "rho_by_hospital_cohort": {
                "H1|c3": 0.18,
                "H2|c3": 0.20,
                "H3|c3": 0.28,
            },
            "scenario_multiplier": 1.03,
        },
        "crisis_guarded": {
            "rho_by_hospital": {"H2": 0.15, "H3": 0.25},
            "rho_by_hospital_cohort": {
                "H3|c3": 0.30,
            },
            "scenario_multiplier": 1.05,
        },
    }

    if profile_name not in profile_map:
        raise ValueError(
            f"Unknown profile_name '{profile_name}'. "
            f"Available: {sorted(profile_map.keys())}"
        )

    cfg = profile_map[profile_name]

    return build_regime_robust_policy_snapshot(
        instance=instance,
        current_regime=current_regime,
        lookahead_horizon=lookahead_horizon,
        use_expected_weights=use_expected_weights,
        rho_by_hospital=cfg["rho_by_hospital"],
        rho_by_hospital_cohort=cfg["rho_by_hospital_cohort"],
        scenario_multiplier=cfg["scenario_multiplier"],
    )


def summarize_regime_vs_robust_policy_snapshot(
    robust_snapshot: dict[str, Any],
    regime_robust_snapshot: dict[str, Any],
) -> pd.DataFrame:
    """
    Lightweight top-level comparison using model_metrics when available.
    """
    robust_metrics = robust_snapshot.get("model_metrics", pd.DataFrame()).copy()
    regime_metrics = regime_robust_snapshot.get("model_metrics", pd.DataFrame()).copy()

    if robust_metrics.empty and regime_metrics.empty:
        return pd.DataFrame()

    if robust_metrics.empty:
        robust_metrics = pd.DataFrame([{"policy_name": "robust_optimized_network"}])

    if regime_metrics.empty:
        regime_metrics = pd.DataFrame([{"policy_name": "regime_robust_optimized_network"}])

    robust_metrics["comparison_side"] = "robust"
    regime_metrics["comparison_side"] = "regime_robust"

    stacked = pd.concat([robust_metrics, regime_metrics], ignore_index=True)

    preferred_cols = [
        "comparison_side",
        "policy_name",
        "current_regime",
        "lookahead_horizon",
        "mean_regime_arrival_multiplier",
        "mean_regime_uncertainty_width",
        "regime_robust_nominal_arrival_total",
        "regime_robust_arrival_total",
        "regime_robust_arrival_delta_total",
        "objective_value",
    ]
    cols = [c for c in preferred_cols if c in stacked.columns]
    if cols:
        return stacked[cols].copy()

    return stacked.copy()