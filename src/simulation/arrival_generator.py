from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config.dataclasses import HealthcareInstance


def build_base_arrivals_table(
    instance: HealthcareInstance,
) -> pd.DataFrame:
    """
    Return the deterministic base arrivals table.

    Expected columns:
        day, hospital_id, cohort, arrivals
    """
    arrivals_df = instance.arrivals.df.copy()

    arrivals_df = arrivals_df[
        ["day", "hospital_id", "cohort", "arrivals"]
    ].sort_values(["day", "hospital_id", "cohort"]).reset_index(drop=True)

    return arrivals_df


def apply_arrival_shock(
    arrivals_df: pd.DataFrame,
    shock_multiplier: float = 1.0,
    cohort: Optional[str] = None,
    hospital_id: Optional[str] = None,
    start_day: Optional[int] = None,
    end_day: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply a multiplicative shock to selected rows of the arrivals table.

    Parameters
    ----------
    shock_multiplier:
        Multiplier applied to matching rows.
    cohort:
        If provided, shock only this cohort.
    hospital_id:
        If provided, shock only this hospital.
    start_day, end_day:
        If provided, shock only within this inclusive day range.
    """
    shocked = arrivals_df.copy()

    mask = pd.Series(True, index=shocked.index)

    if cohort is not None:
        mask &= shocked["cohort"] == cohort

    if hospital_id is not None:
        mask &= shocked["hospital_id"] == hospital_id

    if start_day is not None:
        mask &= shocked["day"] >= int(start_day)

    if end_day is not None:
        mask &= shocked["day"] <= int(end_day)

    shocked.loc[mask, "arrivals"] = (
        shocked.loc[mask, "arrivals"].astype(float) * float(shock_multiplier)
    )

    return shocked


def sample_realized_arrivals(
    arrivals_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Sample realized arrivals from a Poisson model.

    For each row:
        realized_arrivals ~ Poisson(mean = arrivals)

    Returns
    -------
    pd.DataFrame
        Columns:
            day, hospital_id, cohort, arrivals, realized_arrivals
    """
    if rng is None:
        rng = np.random.default_rng()

    sampled = arrivals_df.copy()
    lam = sampled["arrivals"].astype(float).to_numpy()

    sampled["realized_arrivals"] = rng.poisson(lam=lam)
    sampled = sampled.sort_values(
        ["day", "hospital_id", "cohort"]
    ).reset_index(drop=True)

    return sampled


def generate_arrival_scenario(
    instance: HealthcareInstance,
    rng: Optional[np.random.Generator] = None,
    shock_multiplier: float = 1.0,
    cohort: Optional[str] = None,
    hospital_id: Optional[str] = None,
    start_day: Optional[int] = None,
    end_day: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build one realized arrival scenario from the instance.

    Pipeline
    --------
    1. Load deterministic arrivals
    2. Optionally apply a shock
    3. Sample realized arrivals

    Returns
    -------
    pd.DataFrame
        Columns:
            day, hospital_id, cohort, arrivals, realized_arrivals
    """
    base_arrivals = build_base_arrivals_table(instance)

    shocked_arrivals = apply_arrival_shock(
        arrivals_df=base_arrivals,
        shock_multiplier=shock_multiplier,
        cohort=cohort,
        hospital_id=hospital_id,
        start_day=start_day,
        end_day=end_day,
    )

    realized = sample_realized_arrivals(
        arrivals_df=shocked_arrivals,
        rng=rng,
    )

    return realized


def summarize_realized_arrivals(
    realized_arrivals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize realized arrivals by day and hospital.
    """
    summary = (
        realized_arrivals_df.groupby(["day", "hospital_id"], as_index=False)[
            "realized_arrivals"
        ]
        .sum()
        .sort_values(["day", "hospital_id"])
        .reset_index(drop=True)
    )

    return summary