"""Generate a reproducible synthetic historical-arrival replay dataset.

The original MINCO reference case contains a seven-day arrival panel.  That is
sufficient for deterministic Markov projection and scenario optimization, but
it is not sufficient evidence for a temporal machine-learning forecast.  This
module expands the seven-day pattern into a *clearly labelled synthetic* time
series with calendar seasonality, a smooth external pressure signal, bounded
surge episodes, and Poisson count noise.

Nothing produced here is real hospital data.  The generator exists so the
forecasting pipeline can be validated honestly with a genuine time axis while
external historical data remains unavailable.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.dataclasses import HealthcareInstance

DEFAULT_HISTORY_DAYS = 420
DEFAULT_HISTORY_SEED = 20260816
DEFAULT_START_DATE = "2025-01-06"  # Monday; base day 1 maps to Monday.

# Synthetic stress windows are deliberately explicit and reproducible.
DEFAULT_SURGE_WINDOWS: tuple[tuple[int, int], ...] = (
    (55, 70),
    (165, 180),
    (315, 335),
)

_HOSPITAL_SCALE = {"H1": 1.00, "H2": 0.98, "H3": 1.03}


def _in_any_window(day_index: int, windows: Iterable[tuple[int, int]]) -> bool:
    return any(start <= day_index <= end for start, end in windows)


def generate_synthetic_arrival_history(
    instance: HealthcareInstance,
    *,
    n_days: int = DEFAULT_HISTORY_DAYS,
    seed: int = DEFAULT_HISTORY_SEED,
    start_date: str | pd.Timestamp = DEFAULT_START_DATE,
    surge_windows: Iterable[tuple[int, int]] = DEFAULT_SURGE_WINDOWS,
) -> pd.DataFrame:
    """Return a seeded daily arrival history for every hospital/cohort pair.

    Parameters
    ----------
    instance:
        Validated MINCO reference instance.  Its seven-day arrival pattern is
        treated as the day-of-week baseline.
    n_days:
        Number of synthetic historical days to generate.
    seed:
        NumPy random seed recorded by validation manifests.
    start_date:
        Calendar date corresponding to base day 1 (Monday by default).
    surge_windows:
        Inclusive integer day-index windows that increase the external pressure
        signal.  They are scenario-generation assumptions, not observations.
    """
    if n_days < 60:
        raise ValueError("n_days must be at least 60 for lagged temporal validation")

    arrivals = instance.arrivals.df.copy()
    required = {"day", "hospital_id", "cohort", "arrivals"}
    missing = required - set(arrivals.columns)
    if missing:
        raise ValueError(f"Reference arrival table missing columns: {sorted(missing)}")

    base = arrivals.copy()
    base["day_of_week"] = pd.to_numeric(base["day"], errors="raise").astype(int) - 1
    if not base["day_of_week"].between(0, 6).all():
        raise ValueError("Reference arrival day values must map to a seven-day pattern")

    lookup = {
        (int(row.day_of_week), str(row.hospital_id), str(row.cohort)): float(row.arrivals)
        for row in base.itertuples(index=False)
    }
    hospitals = sorted(base["hospital_id"].astype(str).unique().tolist())
    cohorts = sorted(base["cohort"].astype(str).unique().tolist())
    expected_keys = {(dow, h, c) for dow in range(7) for h in hospitals for c in cohorts}
    missing_keys = expected_keys - set(lookup)
    if missing_keys:
        raise ValueError(
            "Reference arrival panel must contain every day/hospital/cohort combination; "
            f"missing {len(missing_keys)} keys"
        )

    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date).normalize()
    rows: list[dict[str, object]] = []

    for day_index in range(n_days):
        date = start + pd.to_timedelta(day_index, unit="D")
        day_of_week = int(date.dayofweek)

        # A smooth exogenous community-pressure proxy plus explicit surge pulses.
        pressure = (
            0.45
            + 0.18 * np.sin(2.0 * np.pi * day_index / 60.0)
            + 0.08 * np.sin(2.0 * np.pi * day_index / 14.0)
        )
        if _in_any_window(day_index, surge_windows):
            pressure += 0.38
        pressure += rng.normal(0.0, 0.035)
        pressure = float(np.clip(pressure, 0.05, 1.0))

        annual_seasonality = 1.0 + 0.08 * np.sin(2.0 * np.pi * day_index / 365.0)

        for hospital_id in hospitals:
            hospital_scale = float(_HOSPITAL_SCALE.get(hospital_id, 1.0))
            for cohort in cohorts:
                baseline = lookup[(day_of_week, hospital_id, cohort)]
                mean_arrivals = (
                    baseline
                    * annual_seasonality
                    * hospital_scale
                    * (0.88 + 0.42 * pressure)
                )
                sampled = int(rng.poisson(max(mean_arrivals, 0.05)))
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "hospital_id": hospital_id,
                        "cohort": cohort,
                        "arrivals": sampled,
                        "community_pressure_index": pressure,
                        "source_mode": "synthetic_historical_replay",
                        "is_synthetic": True,
                        "generator_seed": int(seed),
                    }
                )

    result = pd.DataFrame(rows).sort_values(
        ["date", "hospital_id", "cohort"]
    ).reset_index(drop=True)
    if (result["arrivals"] < 0).any():
        raise AssertionError("Synthetic arrival generator produced a negative count")
    return result


def write_synthetic_arrival_history(
    instance: HealthcareInstance,
    path: Path,
    **kwargs: object,
) -> pd.DataFrame:
    """Generate and persist the reference history as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_arrival_history(instance, **kwargs)
    frame.to_csv(path, index=False)
    return frame
