from __future__ import annotations

from pathlib import Path

from src.config.loader import load_healthcare_instance
from src.markov.forecast_engine import run_forecast
from src.markov.forecast_validation import validate_forecast_against_capacity
from src.metrics.collectors import collect_capacity_metrics


def test_capacity_metric_collector(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    forecast = run_forecast(instance)
    validated = validate_forecast_against_capacity(forecast["resource_demand"], instance)
    metrics = collect_capacity_metrics(validated)
    assert metrics.total_unsafe_excess >= 0
    assert metrics.total_overflow_excess >= 0
    assert metrics.max_utilization_ratio > 0
    assert metrics.num_unsafe_rows >= 0
