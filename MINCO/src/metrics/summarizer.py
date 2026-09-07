from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def summarize_replications(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    metric_columns: Iterable[str],
) -> pd.DataFrame:
    """Return mean, standard deviation, minimum, and maximum by experiment group."""
    groups = list(group_columns)
    metrics = list(metric_columns)
    missing = sorted(set(groups + metrics) - set(frame.columns))
    if missing:
        raise ValueError(f"Replication summary input is missing columns: {missing}")
    summary = frame.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
