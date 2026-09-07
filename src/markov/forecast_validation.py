from __future__ import annotations

import pandas as pd

from src.config.dataclasses import HealthcareInstance


def build_capacity_reference_table(
    instance: HealthcareInstance,
) -> pd.DataFrame:
    """
    Build a unified hospital-resource capacity reference table.

    Returns
    -------
    pd.DataFrame
        Columns:
            hospital_id
            resource
            base_capacity
            safe_utilization
            safe_capacity
            max_surge
            max_capacity_with_surge
    """
    capacities = instance.capacities.df.copy()
    safe_thresholds = instance.safe_thresholds.df.copy()
    surge_caps = instance.surge_caps.df.copy()

    merged = capacities.merge(
        safe_thresholds,
        on=["hospital_id", "resource"],
        how="inner",
        validate="one_to_one",
    ).merge(
        surge_caps,
        on=["hospital_id", "resource"],
        how="inner",
        validate="one_to_one",
    )

    merged["safe_capacity"] = (
        merged["base_capacity"] * merged["safe_utilization"]
    )
    merged["max_capacity_with_surge"] = (
        merged["base_capacity"] + merged["max_surge"]
    )

    return merged[
        [
            "hospital_id",
            "resource",
            "base_capacity",
            "safe_utilization",
            "safe_capacity",
            "max_surge",
            "max_capacity_with_surge",
        ]
    ].copy()


def validate_forecast_against_capacity(
    resource_demand_df: pd.DataFrame,
    instance: HealthcareInstance,
) -> pd.DataFrame:
    """
    Compare forecasted resource demand against safe, base, and surge capacity.

    Parameters
    ----------
    resource_demand_df:
        Output of summarize_resource_demand(...) from forecast_engine.
    instance:
        Loaded healthcare instance.

    Returns
    -------
    pd.DataFrame
        Columns:
            day
            hospital_id
            resource
            expected_demand
            base_capacity
            safe_utilization
            safe_capacity
            max_surge
            max_capacity_with_surge
            unsafe_excess
            overflow_excess
            surge_gap
            utilization_ratio
            safe_utilization_ratio
    """
    capacity_ref = build_capacity_reference_table(instance)

    merged = resource_demand_df.merge(
        capacity_ref,
        on=["hospital_id", "resource"],
        how="left",
        validate="many_to_one",
    )

    if merged[["base_capacity", "safe_capacity", "max_surge"]].isnull().any().any():
        raise ValueError(
            "Some forecast rows could not be matched to capacity reference data."
        )

    merged["unsafe_excess"] = (
        merged["expected_demand"] - merged["safe_capacity"]
    ).clip(lower=0.0)

    merged["overflow_excess"] = (
        merged["expected_demand"] - merged["base_capacity"]
    ).clip(lower=0.0)

    merged["surge_gap"] = (
        merged["expected_demand"] - merged["max_capacity_with_surge"]
    ).clip(lower=0.0)

    merged["utilization_ratio"] = (
        merged["expected_demand"] / merged["base_capacity"]
    )

    merged["safe_utilization_ratio"] = (
        merged["expected_demand"] / merged["safe_capacity"]
    )

    merged = merged.sort_values(
        ["day", "hospital_id", "resource"]
    ).reset_index(drop=True)

    return merged[
        [
            "day",
            "hospital_id",
            "resource",
            "expected_demand",
            "base_capacity",
            "safe_utilization",
            "safe_capacity",
            "max_surge",
            "max_capacity_with_surge",
            "unsafe_excess",
            "overflow_excess",
            "surge_gap",
            "utilization_ratio",
            "safe_utilization_ratio",
        ]
    ].copy()


def summarize_pressure_points(
    validated_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract rows where forecasted demand exceeds safe capacity.
    """
    pressured = validated_forecast_df[
        validated_forecast_df["unsafe_excess"] > 0
    ].copy()

    pressured = pressured.sort_values(
        ["day", "unsafe_excess", "overflow_excess"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return pressured


def summarize_overflow_points(
    validated_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract rows where forecasted demand exceeds base capacity.
    """
    overflow = validated_forecast_df[
        validated_forecast_df["overflow_excess"] > 0
    ].copy()

    overflow = overflow.sort_values(
        ["day", "overflow_excess"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return overflow


def summarize_surge_failure_points(
    validated_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract rows where even full surge capacity would be insufficient.
    """
    failures = validated_forecast_df[
        validated_forecast_df["surge_gap"] > 0
    ].copy()

    failures = failures.sort_values(
        ["day", "surge_gap"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return failures