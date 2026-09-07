from __future__ import annotations

import pandas as pd

from src.metrics.metric_definitions import CapacityMetrics

REQUIRED_COLUMNS = {
    "unsafe_excess",
    "overflow_excess",
    "surge_gap",
    "utilization_ratio",
}


def collect_capacity_metrics(frame: pd.DataFrame) -> CapacityMetrics:
    """Collect canonical pressure metrics from a validated capacity table."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Capacity metric input is missing columns: {missing}")
    if frame.empty:
        return CapacityMetrics(0.0, 0.0, 0.0, 0.0, 0)
    return CapacityMetrics(
        total_unsafe_excess=float(frame["unsafe_excess"].sum()),
        total_overflow_excess=float(frame["overflow_excess"].sum()),
        total_surge_gap=float(frame["surge_gap"].sum()),
        max_utilization_ratio=float(frame["utilization_ratio"].max()),
        num_unsafe_rows=int((frame["unsafe_excess"] > 0).sum()),
    )
