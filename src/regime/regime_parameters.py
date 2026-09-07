from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


DEFAULT_REGIMES = ["normal", "surge", "crisis"]


# ============================================================
# Core regime parameter container
# ============================================================

@dataclass(frozen=True)
class RegimeParameterSet:
    regime: str

    # demand side
    arrival_multiplier: float
    arrival_uncertainty_width: float

    # patient flow side
    icu_discharge_multiplier: float
    ward_discharge_multiplier: float
    stepdown_discharge_multiplier: float

    # operational side
    effective_capacity_multiplier: float
    transfer_capacity_multiplier: float

    # safety side
    safe_threshold_multiplier: float

    # objective side
    overload_penalty_multiplier: float
    blocked_penalty_multiplier: float
    transfer_penalty_multiplier: float

    # optional narrative metadata
    description: str = ""


# ============================================================
# Default calibrated parameters
# ============================================================

def build_default_parameter_table() -> dict[str, RegimeParameterSet]:
    """
    Baseline interpretation:

    normal:
        steady-state operations

    surge:
        higher arrivals, some congestion, slower exits

    crisis:
        extreme arrivals, capacity strain, sticky occupancy
    """

    params = {
        "normal": RegimeParameterSet(
            regime="normal",
            arrival_multiplier=1.00,
            arrival_uncertainty_width=0.08,

            icu_discharge_multiplier=1.00,
            ward_discharge_multiplier=1.00,
            stepdown_discharge_multiplier=1.00,

            effective_capacity_multiplier=1.00,
            transfer_capacity_multiplier=1.00,

            safe_threshold_multiplier=1.00,

            overload_penalty_multiplier=1.00,
            blocked_penalty_multiplier=1.00,
            transfer_penalty_multiplier=1.00,

            description="Stable operating conditions."
        ),

        "surge": RegimeParameterSet(
            regime="surge",
            arrival_multiplier=1.20,
            arrival_uncertainty_width=0.15,

            icu_discharge_multiplier=0.92,
            ward_discharge_multiplier=0.95,
            stepdown_discharge_multiplier=0.95,

            effective_capacity_multiplier=0.97,
            transfer_capacity_multiplier=0.92,

            safe_threshold_multiplier=0.96,

            overload_penalty_multiplier=1.15,
            blocked_penalty_multiplier=1.05,
            transfer_penalty_multiplier=1.08,

            description="Elevated arrivals and moderate congestion."
        ),

        "crisis": RegimeParameterSet(
            regime="crisis",
            arrival_multiplier=1.45,
            arrival_uncertainty_width=0.25,

            icu_discharge_multiplier=0.82,
            ward_discharge_multiplier=0.88,
            stepdown_discharge_multiplier=0.88,

            effective_capacity_multiplier=0.92,
            transfer_capacity_multiplier=0.80,

            safe_threshold_multiplier=0.90,

            overload_penalty_multiplier=1.40,
            blocked_penalty_multiplier=1.12,
            transfer_penalty_multiplier=1.18,

            description="Severe stress with persistent congestion."
        ),
    }

    return params


# ============================================================
# Validation
# ============================================================

def validate_parameter_table(
    parameter_table: dict[str, RegimeParameterSet],
) -> None:
    missing = [r for r in DEFAULT_REGIMES if r not in parameter_table]
    if missing:
        raise ValueError(f"Missing regime parameters for: {missing}")

    for regime, p in parameter_table.items():

        if p.arrival_multiplier <= 0:
            raise ValueError(f"{regime}: arrival_multiplier must be positive.")

        if p.arrival_uncertainty_width < 0:
            raise ValueError(f"{regime}: arrival_uncertainty_width must be nonnegative.")

        for name in [
            "icu_discharge_multiplier",
            "ward_discharge_multiplier",
            "stepdown_discharge_multiplier",
            "effective_capacity_multiplier",
            "transfer_capacity_multiplier",
            "safe_threshold_multiplier",
            "overload_penalty_multiplier",
            "blocked_penalty_multiplier",
            "transfer_penalty_multiplier",
        ]:
            value = getattr(p, name)
            if value <= 0:
                raise ValueError(f"{regime}: {name} must be positive.")


# ============================================================
# Loaders
# ============================================================

def load_default_regime_parameters() -> dict[str, RegimeParameterSet]:
    params = build_default_parameter_table()
    validate_parameter_table(params)
    return params


def get_regime_parameters(
    regime: str,
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> RegimeParameterSet:
    if parameter_table is None:
        parameter_table = load_default_regime_parameters()

    if regime not in parameter_table:
        raise ValueError(
            f"Unknown regime '{regime}'. Valid regimes: {list(parameter_table.keys())}"
        )

    return parameter_table[regime]


# ============================================================
# Convenience extractors
# ============================================================

def arrival_multiplier_map(
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> dict[str, float]:
    if parameter_table is None:
        parameter_table = load_default_regime_parameters()

    return {
        r: p.arrival_multiplier
        for r, p in parameter_table.items()
    }


def uncertainty_width_map(
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> dict[str, float]:
    if parameter_table is None:
        parameter_table = load_default_regime_parameters()

    return {
        r: p.arrival_uncertainty_width
        for r, p in parameter_table.items()
    }


def safe_threshold_map(
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> dict[str, float]:
    if parameter_table is None:
        parameter_table = load_default_regime_parameters()

    return {
        r: p.safe_threshold_multiplier
        for r, p in parameter_table.items()
    }


# ============================================================
# Dynamic scalar helpers
# ============================================================

def regime_adjusted_capacity(
    base_capacity: float,
    regime: str,
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> float:
    p = get_regime_parameters(regime, parameter_table)
    return float(base_capacity * p.effective_capacity_multiplier)


def regime_adjusted_safe_limit(
    nominal_safe_limit: float,
    regime: str,
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> float:
    p = get_regime_parameters(regime, parameter_table)
    return float(nominal_safe_limit * p.safe_threshold_multiplier)


def regime_adjusted_transfer_capacity(
    nominal_transfer_capacity: float,
    regime: str,
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> float:
    p = get_regime_parameters(regime, parameter_table)
    return float(nominal_transfer_capacity * p.transfer_capacity_multiplier)


# ============================================================
# DataFrame views
# ============================================================

def regime_parameter_dataframe(
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> pd.DataFrame:
    if parameter_table is None:
        parameter_table = load_default_regime_parameters()

    rows = []
    for regime, p in parameter_table.items():
        row = asdict(p)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Optimization-facing helpers
# ============================================================

def expected_weighted_parameter(
    weights: dict[str, float],
    field_name: str,
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> float:
    """
    Example:
        weights = {"normal":0.7, "surge":0.2, "crisis":0.1}
        field_name = "arrival_multiplier"
    """
    if parameter_table is None:
        parameter_table = load_default_regime_parameters()

    total = 0.0

    for regime, w in weights.items():
        p = get_regime_parameters(regime, parameter_table)
        total += float(w) * float(getattr(p, field_name))

    return total


def expected_weighted_arrival_multiplier(
    weights: dict[str, float],
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> float:
    return expected_weighted_parameter(
        weights=weights,
        field_name="arrival_multiplier",
        parameter_table=parameter_table,
    )


def expected_weighted_uncertainty_width(
    weights: dict[str, float],
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> float:
    return expected_weighted_parameter(
        weights=weights,
        field_name="arrival_uncertainty_width",
        parameter_table=parameter_table,
    )


# ============================================================
# Scenario override utility
# ============================================================

def clone_with_override(
    regime: str,
    parameter_table: dict[str, RegimeParameterSet],
    **kwargs: Any,
) -> dict[str, RegimeParameterSet]:
    """
    Example:
        new_params = clone_with_override(
            regime="crisis",
            parameter_table=params,
            arrival_multiplier=1.60
        )
    """
    if regime not in parameter_table:
        raise ValueError(f"Unknown regime '{regime}'.")

    old = parameter_table[regime]
    data = asdict(old)

    for k, v in kwargs.items():
        if k not in data:
            raise ValueError(f"Invalid field '{k}'.")
        data[k] = v

    new_table = dict(parameter_table)
    new_table[regime] = RegimeParameterSet(**data)

    validate_parameter_table(new_table)
    return new_table


# ============================================================
# Pretty printer
# ============================================================

def print_regime_parameter_summary(
    parameter_table: dict[str, RegimeParameterSet] | None = None,
) -> None:
    if parameter_table is None:
        parameter_table = load_default_regime_parameters()

    df = regime_parameter_dataframe(parameter_table)

    print("\nREGIME PARAMETER SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))