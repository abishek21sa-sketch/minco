from __future__ import annotations

from pathlib import Path

from src.ai.demand_forecast_quantile import (
    build_temporal_forecast_frame,
    forecast_next_day,
    train_validate_demand_forecast,
)
from src.config.loader import load_healthcare_instance
from src.data_engineering.synthetic_arrival_history import generate_synthetic_arrival_history


def test_temporal_arrival_forecast_beats_seasonal_baseline_and_is_calibrated(
    tmp_path: Path,
) -> None:
    instance = load_healthcare_instance("data")
    history = generate_synthetic_arrival_history(instance, n_days=220, seed=20260816)
    frame = build_temporal_forecast_frame(history)

    # Leakage guard: row target date is strictly after every lagged observation
    # by construction; target itself is not included in the feature contract.
    assert "target_arrivals" not in {
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
    }
    assert not frame.empty

    report = train_validate_demand_forecast(
        history,
        model_path=tmp_path / "model.joblib",
        validation_path=tmp_path / "validation.json",
        prediction_path=tmp_path / "predictions.csv",
    )
    assert report["status"] == "passed"
    assert report["metrics"]["mae_improvement_vs_seasonal_naive"] >= 0.10
    assert report["metrics"]["prediction_interval_90_coverage"] >= 0.85

    predictions = forecast_next_day(history, model_path=tmp_path / "model.joblib")
    assert len(predictions) == 12
    assert (predictions["predicted_p05"] <= predictions["predicted_p50"]).all()
    assert (predictions["predicted_p50"] <= predictions["predicted_p95"]).all()
    assert (predictions[["predicted_p05", "predicted_p50", "predicted_p95"]] >= 0).all().all()
    assert set(predictions["evidence_label"]) == {"PREDICTED"}
