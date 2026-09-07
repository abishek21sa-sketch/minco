from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.dataclasses import HealthcareInstance


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"None of the candidate columns {candidates} found in dataframe columns {list(df.columns)}"
    )


def _row_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def _clone_data_wrapper(original_wrapper: Any, new_df: pd.DataFrame) -> Any:
    wrapper_cls = type(original_wrapper)
    return wrapper_cls(df=new_df)


def _apply_arrival_config(arrivals_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if arrivals_df.empty:
        return arrivals_df

    df = arrivals_df.copy()

    hospital_col = _find_first_existing(df, ["hospital_id", "hospital"])
    cohort_col = _find_first_existing(df, ["cohort", "patient_cohort"])
    arrival_col = _find_first_existing(
        df,
        ["arrival_rate", "arrivals", "lambda", "arrival_volume", "mean_arrivals"],
    )

    arrivals_cfg = cfg.get("arrivals", {})
    default_multiplier = float(arrivals_cfg.get("default_multiplier", 1.0))
    by_hospital_cohort = arrivals_cfg.get("by_hospital_cohort", {})

    df[arrival_col] = pd.to_numeric(df[arrival_col], errors="coerce").fillna(0.0)
    df[arrival_col] = df[arrival_col] * default_multiplier

    def row_multiplier(row: pd.Series) -> float:
        key = _row_key(str(row[hospital_col]), str(row[cohort_col]))
        return float(by_hospital_cohort.get(key, 1.0))

    df[arrival_col] = df[arrival_col] * df.apply(row_multiplier, axis=1)
    return df


def _apply_capacity_config(
    capacities_df: pd.DataFrame,
    safe_df: pd.DataFrame,
    surge_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cap_df = capacities_df.copy()
    safe_out = safe_df.copy()
    surge_out = surge_df.copy()

    cap_cfg = cfg.get("capacities", {})

    base_overrides = cap_cfg.get("base_capacity_overrides", {})
    safe_ratio_default = float(cap_cfg.get("safe_capacity_ratio", 0.9))
    safe_overrides = cap_cfg.get("safe_capacity_overrides", {})
    surge_overrides = cap_cfg.get("max_surge_overrides", {})

    cap_h_col = _find_first_existing(cap_df, ["hospital_id", "hospital"])
    cap_r_col = _find_first_existing(cap_df, ["resource", "unit", "unit_type"])
    cap_val_col = _find_first_existing(cap_df, ["base_capacity", "capacity"])

    cap_df[cap_val_col] = pd.to_numeric(cap_df[cap_val_col], errors="coerce").fillna(0.0)

    for idx, row in cap_df.iterrows():
        key = _row_key(str(row[cap_h_col]), str(row[cap_r_col]))
        if key in base_overrides:
            cap_df.at[idx, cap_val_col] = float(base_overrides[key])

    base_lookup = {
        _row_key(str(r[cap_h_col]), str(r[cap_r_col])): float(r[cap_val_col])
        for _, r in cap_df.iterrows()
    }

    safe_h_col = _find_first_existing(safe_out, ["hospital_id", "hospital"])
    safe_r_col = _find_first_existing(safe_out, ["resource", "unit", "unit_type"])

    if "safe_capacity" in safe_out.columns:
        safe_val_col = "safe_capacity"
        safe_out[safe_val_col] = pd.to_numeric(safe_out[safe_val_col], errors="coerce").fillna(0.0)

        for idx, row in safe_out.iterrows():
            key = _row_key(str(row[safe_h_col]), str(row[safe_r_col]))
            if key in safe_overrides:
                safe_out.at[idx, safe_val_col] = float(safe_overrides[key])
            elif key in base_lookup:
                safe_out.at[idx, safe_val_col] = float(base_lookup[key] * safe_ratio_default)

    elif "safe_utilization" in safe_out.columns:
        safe_val_col = "safe_utilization"
        safe_out[safe_val_col] = pd.to_numeric(safe_out[safe_val_col], errors="coerce").fillna(
            safe_ratio_default
        )

        for idx, row in safe_out.iterrows():
            key = _row_key(str(row[safe_h_col]), str(row[safe_r_col]))
            if key in safe_overrides and key in base_lookup and base_lookup[key] > 0:
                safe_out.at[idx, safe_val_col] = float(safe_overrides[key]) / float(
                    base_lookup[key]
                )
            elif key not in safe_overrides:
                safe_out.at[idx, safe_val_col] = safe_ratio_default

    else:
        raise ValueError(
            f"Safe-threshold dataframe must contain either 'safe_capacity' or 'safe_utilization'. "
            f"Found columns: {list(safe_out.columns)}"
        )

    surge_h_col = _find_first_existing(surge_out, ["hospital_id", "hospital"])
    surge_r_col = _find_first_existing(surge_out, ["resource", "unit", "unit_type"])
    surge_val_col = _find_first_existing(
        surge_out,
        ["max_surge", "surge_capacity", "extra_capacity"],
    )

    surge_out[surge_val_col] = pd.to_numeric(surge_out[surge_val_col], errors="coerce").fillna(0.0)

    for idx, row in surge_out.iterrows():
        key = _row_key(str(row[surge_h_col]), str(row[surge_r_col]))
        if key in surge_overrides:
            surge_out.at[idx, surge_val_col] = float(surge_overrides[key])

    return cap_df, safe_out, surge_out


def _apply_transfer_config(transfer_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if transfer_df.empty:
        return transfer_df

    df = transfer_df.copy()

    from_col = _find_first_existing(df, ["from_hospital", "origin_hospital", "from"])
    to_col = _find_first_existing(df, ["to_hospital", "destination_hospital", "to"])

    transfer_cfg = cfg.get("transfer_lanes", {})
    default_allowed = transfer_cfg.get("default_allowed", None)
    lane_overrides = transfer_cfg.get("lane_overrides", {})

    value_col = None
    for c in ["capacity", "max_transfer", "transfer_capacity", "allowed"]:
        if c in df.columns:
            value_col = c
            break

    if value_col is None:
        df["allowed"] = 1
        value_col = "allowed"

    if default_allowed is not None:
        df[value_col] = default_allowed

    for idx, row in df.iterrows():
        key = _row_key(str(row[from_col]), str(row[to_col]))
        if key in lane_overrides:
            df.at[idx, value_col] = lane_overrides[key]

    return df


def apply_parameter_config(
    instance: HealthcareInstance,
    config_path: str | Path,
) -> HealthcareInstance:
    cfg = _load_json(config_path)

    arrivals_df = _apply_arrival_config(instance.arrivals.df, cfg)

    capacities_df, safe_df, surge_df = _apply_capacity_config(
        instance.capacities.df,
        instance.safe_thresholds.df,
        instance.surge_caps.df,
        cfg,
    )

    transfer_df = _apply_transfer_config(instance.transfer_lanes.df, cfg)

    # Leave transitions untouched for now because this project stores them
    # in a TransitionBundle rather than a simple df wrapper.
    transitions_obj = instance.transitions

    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=_clone_data_wrapper(instance.capacities, capacities_df),
        surge_caps=_clone_data_wrapper(instance.surge_caps, surge_df),
        safe_thresholds=_clone_data_wrapper(instance.safe_thresholds, safe_df),
        transfer_lanes=_clone_data_wrapper(instance.transfer_lanes, transfer_df),
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=_clone_data_wrapper(instance.arrivals, arrivals_df),
        transitions=transitions_obj,
    )
