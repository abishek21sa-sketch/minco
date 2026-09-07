"""Discrete-time discharge hazard model for operational bed-release forecasting.

This is intentionally an operational survival model rather than a generic LOS
regression.  A patient stay is expanded into time bins and logistic regression
estimates the conditional probability of discharge in each bin given survival
through the previous bin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DiscreteTimeHazardModel:
    bin_hours: int = 6
    max_horizon_hours: int = 120
    numeric_features: tuple[str, ...] = ("age", "acuity_score")
    categorical_features: tuple[str, ...] = ("unit", "cohort")
    model: Pipeline | None = None

    def _build_model(self) -> Pipeline:
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocess = ColumnTransformer([
            ("numeric", numeric, list(self.numeric_features) + ["time_bin"]),
            ("categorical", categorical, list(self.categorical_features)),
        ])
        return Pipeline([
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])

    def fit(self, stays: pd.DataFrame) -> "DiscreteTimeHazardModel":
        person_period = expand_stays_to_person_period(
            stays,
            bin_hours=self.bin_hours,
            max_horizon_hours=self.max_horizon_hours,
            feature_columns=list(self.numeric_features + self.categorical_features),
        )
        if person_period["discharged_in_bin"].nunique() < 2:
            raise ValueError("Hazard training requires both discharge and non-discharge rows")
        self.model = self._build_model()
        x_cols = list(self.numeric_features + self.categorical_features) + ["time_bin"]
        self.model.fit(person_period[x_cols], person_period["discharged_in_bin"])
        return self

    def _require_model(self) -> Pipeline:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model

    def predict_hazard_curve(self, patients: pd.DataFrame, horizons_hours: Sequence[int]) -> pd.DataFrame:
        model = self._require_model()
        if not horizons_hours:
            raise ValueError("At least one horizon is required")
        horizons = sorted({int(h) for h in horizons_hours})
        if horizons[0] <= 0 or horizons[-1] > self.max_horizon_hours:
            raise ValueError("Horizons must be within the configured model horizon")
        max_bin = int(np.ceil(horizons[-1] / self.bin_hours))
        base = patients.reset_index(drop=True).copy()
        required = set(self.numeric_features + self.categorical_features)
        missing = required - set(base.columns)
        if missing:
            raise ValueError(f"Patient frame missing columns: {sorted(missing)}")
        n = len(base)
        if n == 0:
            return pd.DataFrame(columns=["patient_row", "horizon_hours", "discharge_probability", "survival_probability"])

        repeated = base.loc[np.repeat(base.index.to_numpy(), max_bin)].reset_index(drop=True)
        repeated["time_bin"] = np.tile(np.arange(1, max_bin + 1), n)
        x_cols = list(self.numeric_features + self.categorical_features) + ["time_bin"]
        hazards = model.predict_proba(repeated[x_cols])[:, 1].reshape(n, max_bin)
        hazards = np.clip(hazards, 0.0, 1.0)
        survival = np.cumprod(1.0 - hazards, axis=1)
        cumulative = 1.0 - survival

        records: list[pd.DataFrame] = []
        patient_rows = np.arange(n)
        for horizon in horizons:
            bin_index = int(np.ceil(horizon / self.bin_hours)) - 1
            prob = cumulative[:, bin_index]
            records.append(pd.DataFrame({
                "patient_row": patient_rows,
                "horizon_hours": horizon,
                "discharge_probability": prob,
                "survival_probability": 1.0 - prob,
            }))
        return pd.concat(records, ignore_index=True)


def expand_stays_to_person_period(
    stays: pd.DataFrame,
    *,
    bin_hours: int,
    max_horizon_hours: int,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    required = {"duration_hours", "event_observed", *feature_columns}
    missing = required - set(stays.columns)
    if missing:
        raise ValueError(f"Stay data missing columns: {sorted(missing)}")
    if bin_hours <= 0 or max_horizon_hours <= 0:
        raise ValueError("bin_hours and max_horizon_hours must be positive")

    rows: list[dict[str, object]] = []
    max_bins = int(np.ceil(max_horizon_hours / bin_hours))
    for patient_index, stay in stays.reset_index(drop=True).iterrows():
        duration = float(stay["duration_hours"])
        if duration <= 0:
            raise ValueError("duration_hours must be positive")
        event = bool(stay["event_observed"])
        observed_bin = max(1, int(np.ceil(duration / bin_hours)))
        last_bin = min(observed_bin, max_bins)
        for time_bin in range(1, last_bin + 1):
            discharged = int(event and observed_bin == time_bin and observed_bin <= max_bins)
            record = {
                "patient_row": patient_index,
                "time_bin": time_bin,
                "discharged_in_bin": discharged,
            }
            for feature in feature_columns:
                record[feature] = stay[feature]
            rows.append(record)
    return pd.DataFrame(rows)


def synthetic_stay_dataset(n_patients: int = 1200, *, seed: int = 20260817) -> pd.DataFrame:
    """Generate a reproducible synthetic stay cohort for algorithm validation only."""
    if n_patients < 100:
        raise ValueError("At least 100 patients are required")
    rng = np.random.default_rng(seed)
    units = rng.choice(["Ward", "ICU", "Stepdown"], size=n_patients, p=[0.65, 0.18, 0.17])
    cohorts = rng.choice(["medical", "surgical", "respiratory", "cardiac"], size=n_patients)
    age = np.clip(rng.normal(62, 17, size=n_patients), 18, 95)
    acuity = np.clip(rng.normal(0.45, 0.20, size=n_patients), 0.02, 0.98)
    unit_multiplier = np.where(units == "ICU", 1.9, np.where(units == "Stepdown", 1.25, 1.0))
    cohort_multiplier = np.where(cohorts == "respiratory", 1.25, np.where(cohorts == "cardiac", 1.15, 1.0))
    mean_hours = (38 + 36 * acuity + 0.22 * (age - 50)) * unit_multiplier * cohort_multiplier
    durations = np.maximum(3.0, rng.gamma(shape=2.4, scale=mean_hours / 2.4))
    censor_time = rng.uniform(72, 144, size=n_patients)
    observed = durations <= censor_time
    observed_duration = np.minimum(durations, censor_time)
    return pd.DataFrame({
        "duration_hours": observed_duration,
        "event_observed": observed.astype(int),
        "age": age,
        "acuity_score": acuity,
        "unit": units,
        "cohort": cohorts,
    })
