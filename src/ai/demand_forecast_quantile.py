"""Synthetic-benchmark probabilistic next-day arrival forecasting for MINCO.

This is the first MINCO forecasting component with a genuine operational time
axis.  Earlier research artifacts ordered independent Monte Carlo replications
and therefore could not support a defensible temporal forecast claim.

The model here predicts next-day arrivals for each hospital/cohort using only
information available before the forecast date.  Validation is chronological,
uses a seasonal-naive baseline, and reports a conformalized 90% prediction
interval.  All evidence remains synthetic-reference-case evidence.
"""
from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.paths import PROJECT_ROOT, RESULTS_DIR
from src.validation.run_manifest import sha256_file
from src.version import __version__

RANDOM_STATE = 20260816
TARGET_COVERAGE = 0.90
LOWER_QUANTILE = 0.05
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = 0.95

FEATURES: tuple[str, ...] = (
    "hospital_id",
    "cohort",
    "forecast_day_of_week",
    "forecast_month",
    "community_pressure_index_lag1",
    "time_index",
    "arrivals_lag1",
    "arrivals_lag7",
    "arrivals_roll7",
    "arrivals_roll28",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("hospital_id", "cohort")
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES
)

DEFAULT_MODEL_PATH = RESULTS_DIR / "model_registry" / "models" / "next_day_arrivals_quantile.joblib"
DEFAULT_VALIDATION_PATH = RESULTS_DIR / "validation" / "demand_forecast_validation.json"
DEFAULT_PREDICTION_PATH = RESULTS_DIR / "validation" / "demand_forecast_test_predictions.csv"


@dataclass(frozen=True)
class TemporalSplit:
    train_end: str
    calibration_end: str
    n_train: int
    n_calibration: int
    n_test: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_end": self.train_end,
            "calibration_end": self.calibration_end,
            "n_train": self.n_train,
            "n_calibration": self.n_calibration,
            "n_test": self.n_test,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_temporal_forecast_frame(history_df: pd.DataFrame) -> pd.DataFrame:
    """Build forecast-date rows using only lagged/past information."""
    required = {
        "date",
        "hospital_id",
        "cohort",
        "arrivals",
        "community_pressure_index",
    }
    missing = required - set(history_df.columns)
    if missing:
        raise ValueError(f"Arrival history missing columns: {sorted(missing)}")

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["arrivals"] = pd.to_numeric(df["arrivals"], errors="raise").astype(float)
    df["community_pressure_index"] = pd.to_numeric(
        df["community_pressure_index"], errors="raise"
    ).astype(float)
    if (df["arrivals"] < 0).any():
        raise ValueError("Arrival history contains negative counts")

    df = df.sort_values(["hospital_id", "cohort", "date"]).reset_index(drop=True)
    groups = ["hospital_id", "cohort"]
    start_date = df["date"].min()

    # Each row's date is the forecast/target date.  All predictor values below
    # are shifted so they are available no later than the previous day.
    df["forecast_day_of_week"] = df["date"].dt.dayofweek.astype(int)
    df["forecast_month"] = df["date"].dt.month.astype(int)
    df["time_index"] = (df["date"] - start_date).dt.days.astype(int)
    df["community_pressure_index_lag1"] = df.groupby(groups)[
        "community_pressure_index"
    ].shift(1)
    df["arrivals_lag1"] = df.groupby(groups)["arrivals"].shift(1)
    df["arrivals_lag7"] = df.groupby(groups)["arrivals"].shift(7)
    df["arrivals_roll7"] = df.groupby(groups)["arrivals"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).mean()
    )
    df["arrivals_roll28"] = df.groupby(groups)["arrivals"].transform(
        lambda series: series.shift(1).rolling(28, min_periods=14).mean()
    )
    df["target_arrivals"] = df["arrivals"]
    df["seasonal_naive"] = df["arrivals_lag7"]

    frame = df.dropna(subset=list(FEATURES) + ["target_arrivals", "seasonal_naive"]).copy()
    return frame.reset_index(drop=True)


def chronological_train_calibration_test(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TemporalSplit]:
    """Split by unique forecast date with no temporal overlap."""
    if not 0.5 <= train_fraction < 0.9:
        raise ValueError("train_fraction must be in [0.5, 0.9)")
    if not 0.05 <= calibration_fraction <= 0.3:
        raise ValueError("calibration_fraction must be in [0.05, 0.3]")
    if train_fraction + calibration_fraction >= 0.95:
        raise ValueError("At least 5% of dates must remain for testing")

    dates = sorted(pd.to_datetime(frame["date"]).dt.normalize().unique())
    if len(dates) < 20:
        raise ValueError("At least 20 unique dates are required for temporal validation")

    train_cut = max(1, int(math.floor(len(dates) * train_fraction)))
    calibration_cut = max(train_cut + 1, int(math.floor(len(dates) * (train_fraction + calibration_fraction))))
    calibration_cut = min(calibration_cut, len(dates) - 1)
    train_dates = set(dates[:train_cut])
    calibration_dates = set(dates[train_cut:calibration_cut])
    test_dates = set(dates[calibration_cut:])

    date_series = pd.to_datetime(frame["date"]).dt.normalize()
    train = frame[date_series.isin(train_dates)].copy()
    calibration = frame[date_series.isin(calibration_dates)].copy()
    test = frame[date_series.isin(test_dates)].copy()
    if train.empty or calibration.empty or test.empty:
        raise AssertionError("Temporal split produced an empty partition")
    if pd.to_datetime(train["date"]).max() >= pd.to_datetime(calibration["date"]).min():
        raise AssertionError("Training and calibration periods overlap")
    if pd.to_datetime(calibration["date"]).max() >= pd.to_datetime(test["date"]).min():
        raise AssertionError("Calibration and test periods overlap")

    split = TemporalSplit(
        train_end=pd.Timestamp(pd.to_datetime(train["date"]).max()).date().isoformat(),
        calibration_end=pd.Timestamp(pd.to_datetime(calibration["date"]).max()).date().isoformat(),
        n_train=len(train),
        n_calibration=len(calibration),
        n_test=len(test),
    )
    return train, calibration, test, split


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=0.0,
    )


def _quantile_model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=alpha,
                    random_state=RANDOM_STATE,
                    n_estimators=220,
                    max_depth=2,
                    learning_rate=0.04,
                ),
            ),
        ]
    )


def _conformal_delta(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    target_coverage: float = TARGET_COVERAGE,
) -> float:
    """Conformalize a central interval using a held-out calibration set."""
    scores = np.maximum(lower - y_true, y_true - upper)
    alpha = 1.0 - target_coverage
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration set is empty")
    quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(max(0.0, np.quantile(scores, quantile_level, method="higher")))


def train_validate_demand_forecast(
    history_df: pd.DataFrame,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    prediction_path: Path = DEFAULT_PREDICTION_PATH,
) -> dict[str, Any]:
    """Train, validate, persist, and govern the probabilistic arrival model."""
    frame = build_temporal_forecast_frame(history_df)
    train, calibration, test, split = chronological_train_calibration_test(frame)

    models = {
        "p05": _quantile_model(LOWER_QUANTILE),
        "p50": _quantile_model(MEDIAN_QUANTILE),
        "p95": _quantile_model(UPPER_QUANTILE),
    }
    for model in models.values():
        model.fit(train[list(FEATURES)], train["target_arrivals"])

    calibration_lower = np.clip(models["p05"].predict(calibration[list(FEATURES)]), 0.0, None)
    calibration_upper = np.clip(models["p95"].predict(calibration[list(FEATURES)]), 0.0, None)
    delta = _conformal_delta(
        calibration["target_arrivals"].to_numpy(dtype=float),
        calibration_lower,
        calibration_upper,
    )

    test_lower_raw = np.clip(models["p05"].predict(test[list(FEATURES)]), 0.0, None)
    test_median = np.clip(models["p50"].predict(test[list(FEATURES)]), 0.0, None)
    test_upper_raw = np.clip(models["p95"].predict(test[list(FEATURES)]), 0.0, None)
    test_lower = np.clip(test_lower_raw - delta, 0.0, None)
    test_upper = np.maximum(test_upper_raw + delta, test_lower)
    y_test = test["target_arrivals"].to_numpy(dtype=float)
    seasonal_naive = test["seasonal_naive"].to_numpy(dtype=float)

    model_mae = float(mean_absolute_error(y_test, test_median))
    baseline_mae = float(mean_absolute_error(y_test, seasonal_naive))
    improvement = float((baseline_mae - model_mae) / baseline_mae) if baseline_mae > 0 else 0.0
    rmse = float(math.sqrt(mean_squared_error(y_test, test_median)))
    coverage = float(np.mean((y_test >= test_lower) & (y_test <= test_upper)))
    mean_width = float(np.mean(test_upper - test_lower))

    checks = {
        "chronological_partitions_non_overlapping": True,
        "beats_seasonal_naive_by_at_least_10_percent": improvement >= 0.10,
        "prediction_interval_coverage_at_least_85_percent": coverage >= 0.85,
        "prediction_interval_coverage_not_degenerate": coverage <= 0.995,
        "nonnegative_predictions": bool(
            (test_lower >= 0).all() and (test_median >= 0).all() and (test_upper >= 0).all()
        ),
    }
    status = "passed" if all(checks.values()) else "failed"

    model_path = Path(model_path)
    validation_path = Path(validation_path)
    prediction_path = Path(prediction_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "artifact_type": "minco_probabilistic_arrival_forecast",
        "artifact_version": 1,
        "created_at": _utc_now(),
        "synthetic_reference_case_only": True,
        "features": list(FEATURES),
        "models": models,
        "conformal_delta": delta,
        "target_coverage": TARGET_COVERAGE,
        "history_start": str(pd.to_datetime(history_df["date"]).min().date()),
        "history_end": str(pd.to_datetime(history_df["date"]).max().date()),
        "split": split.to_dict(),
        "metrics": {
            "test_mae_p50": model_mae,
            "test_rmse_p50": rmse,
            "seasonal_naive_mae": baseline_mae,
            "mae_improvement_vs_seasonal_naive": improvement,
            "prediction_interval_90_coverage": coverage,
            "prediction_interval_90_mean_width": mean_width,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "minco": __version__,
        },
    }
    joblib.dump(bundle, model_path)
    model_sha256 = sha256_file(model_path)

    predictions = test[
        ["date", "hospital_id", "cohort", "target_arrivals", "seasonal_naive"]
    ].copy()
    predictions["predicted_p05"] = test_lower
    predictions["predicted_p50"] = test_median
    predictions["predicted_p95"] = test_upper
    predictions["evidence_label"] = "PREDICTED_SYNTHETIC_VALIDATION"
    predictions.to_csv(prediction_path, index=False)

    report: dict[str, Any] = {
        "status": status,
        "method": "chronological_quantile_gradient_boosting_with_conformal_interval",
        "evidence_basis": "synthetic_historical_replay",
        "claim_boundary": (
            "This validates next-day arrival forecasting only on the generated MINCO synthetic "
            "historical replay. It is not evidence of real-hospital predictive accuracy."
        ),
        "feature_contract": list(FEATURES),
        "target": "next_day_hospital_cohort_arrivals",
        "baseline": "seasonal_naive_same_weekday",
        "split": split.to_dict(),
        "metrics": bundle["metrics"],
        "checks": checks,
        "model_path": _relative(model_path),
        "model_sha256": model_sha256,
        "test_predictions_path": _relative(prediction_path),
    }
    validation_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_demand_forecast_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Demand forecast model not found at {path}. Run the finalization Phase 1 gate first."
        )
    bundle = joblib.load(path)
    if bundle.get("artifact_type") != "minco_probabilistic_arrival_forecast":
        raise ValueError("Unexpected demand forecast artifact type")
    return bundle


def build_next_day_feature_rows(history_df: pd.DataFrame) -> pd.DataFrame:
    """Build one leakage-safe next-day feature row per hospital/cohort."""
    required = {
        "date",
        "hospital_id",
        "cohort",
        "arrivals",
        "community_pressure_index",
    }
    missing = required - set(history_df.columns)
    if missing:
        raise ValueError(f"Arrival history missing columns: {sorted(missing)}")

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["arrivals"] = pd.to_numeric(df["arrivals"], errors="raise").astype(float)
    df = df.sort_values(["hospital_id", "cohort", "date"]).reset_index(drop=True)
    start_date = df["date"].min().normalize()
    latest_date = df["date"].max().normalize()
    forecast_date = latest_date + pd.Timedelta(value=1, unit="D")

    rows: list[dict[str, Any]] = []
    for (hospital_id, cohort), group in df.groupby(["hospital_id", "cohort"], sort=True):
        group = group.sort_values("date")
        if len(group) < 28:
            raise ValueError("At least 28 historical observations are required per hospital/cohort")
        arrivals = group["arrivals"].to_numpy(dtype=float)
        pressure = pd.to_numeric(group["community_pressure_index"], errors="raise").to_numpy(dtype=float)
        rows.append(
            {
                "forecast_date": forecast_date.date().isoformat(),
                "hospital_id": str(hospital_id),
                "cohort": str(cohort),
                "forecast_day_of_week": int(forecast_date.dayofweek),
                "forecast_month": int(forecast_date.month),
                "community_pressure_index_lag1": float(pressure[-1]),
                "time_index": int((forecast_date - start_date).days),
                "arrivals_lag1": float(arrivals[-1]),
                "arrivals_lag7": float(arrivals[-7]),
                "arrivals_roll7": float(np.mean(arrivals[-7:])),
                "arrivals_roll28": float(np.mean(arrivals[-28:])),
                "seasonal_naive": float(arrivals[-7]),
            }
        )
    return pd.DataFrame(rows)


def forecast_next_day(
    history_df: pd.DataFrame,
    *,
    bundle: dict[str, Any] | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Generate p05/p50/p95 next-day arrival forecasts by hospital/cohort."""
    bundle = bundle or load_demand_forecast_bundle(model_path)
    rows = build_next_day_feature_rows(history_df)
    features = bundle.get("features", list(FEATURES))
    x = rows[features]
    models = bundle["models"]
    delta = float(bundle.get("conformal_delta", 0.0))

    lower = np.clip(models["p05"].predict(x) - delta, 0.0, None)
    median = np.clip(models["p50"].predict(x), 0.0, None)
    upper = np.clip(models["p95"].predict(x) + delta, 0.0, None)
    upper = np.maximum(upper, median)
    lower = np.minimum(lower, median)

    result = rows[["forecast_date", "hospital_id", "cohort", "seasonal_naive"]].copy()
    result["predicted_p05"] = lower
    result["predicted_p50"] = median
    result["predicted_p95"] = upper
    result["evidence_label"] = "PREDICTED"
    result["validation_scope"] = "synthetic_reference_case_only"
    return result.sort_values(["hospital_id", "cohort"]).reset_index(drop=True)
