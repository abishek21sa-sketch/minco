"""Time-respecting and group-aware validation splits for operational forecasts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class SplitMetadata:
    strategy: str
    n_train: int
    n_test: int
    train_time_min: float
    train_time_max: float
    test_time_min: float
    test_time_max: float
    group_columns: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_fraction(test_fraction: float) -> None:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")


def _metadata(
    df: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
    *,
    strategy: str,
    time_col: str,
    group_cols: Sequence[str] = (),
) -> SplitMetadata:
    if train_index.empty or test_index.empty:
        raise ValueError("Both train and test partitions must be non-empty")
    return SplitMetadata(
        strategy=strategy,
        n_train=len(train_index),
        n_test=len(test_index),
        train_time_min=float(df.loc[train_index, time_col].min()),
        train_time_max=float(df.loc[train_index, time_col].max()),
        test_time_min=float(df.loc[test_index, time_col].min()),
        test_time_max=float(df.loc[test_index, time_col].max()),
        group_columns=tuple(group_cols),
    )


def chronological_holdout(
    df: pd.DataFrame,
    *,
    time_col: str = "time_index",
    test_fraction: float = 0.25,
) -> tuple[pd.Index, pd.Index, SplitMetadata]:
    """Hold out the latest unique time periods across the entire dataset."""
    _validate_fraction(test_fraction)
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    unique_times = sorted(pd.Series(df[time_col].dropna().unique()).tolist())
    if len(unique_times) < 2:
        raise ValueError("At least two unique time values are required")
    n_test_times = max(1, int(round(len(unique_times) * test_fraction)))
    n_test_times = min(n_test_times, len(unique_times) - 1)
    first_test_time = unique_times[-n_test_times]

    train_index = df.index[df[time_col] < first_test_time]
    test_index = df.index[df[time_col] >= first_test_time]
    metadata = _metadata(
        df,
        train_index,
        test_index,
        strategy="chronological_holdout",
        time_col=time_col,
    )
    if metadata.train_time_max >= metadata.test_time_min:
        raise AssertionError("Chronological split contains temporal overlap")
    return train_index, test_index, metadata


def grouped_chronological_holdout(
    df: pd.DataFrame,
    *,
    group_cols: Iterable[str],
    time_col: str = "time_index",
    test_fraction: float = 0.25,
) -> tuple[pd.Index, pd.Index, SplitMetadata]:
    """Hold out the final time window independently inside every trajectory group."""
    _validate_fraction(test_fraction)
    group_cols = tuple(col for col in group_cols if col in df.columns)
    if not group_cols:
        raise ValueError("At least one valid group column is required")
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    train_labels: list[int] = []
    test_labels: list[int] = []
    for _, group in df.groupby(list(group_cols), dropna=False, sort=False):
        unique_times = sorted(pd.Series(group[time_col].dropna().unique()).tolist())
        if len(unique_times) < 2:
            continue
        n_test_times = max(1, int(round(len(unique_times) * test_fraction)))
        n_test_times = min(n_test_times, len(unique_times) - 1)
        first_test_time = unique_times[-n_test_times]
        train_labels.extend(group.index[group[time_col] < first_test_time].tolist())
        test_labels.extend(group.index[group[time_col] >= first_test_time].tolist())

    train_index = pd.Index(sorted(set(train_labels)))
    test_index = pd.Index(sorted(set(test_labels)))
    if train_index.intersection(test_index).size:
        raise AssertionError("Grouped temporal split contains overlapping rows")

    metadata = _metadata(
        df,
        train_index,
        test_index,
        strategy="grouped_chronological_holdout",
        time_col=time_col,
        group_cols=group_cols,
    )
    return train_index, test_index, metadata


def leave_one_group_out(
    df: pd.DataFrame,
    *,
    group_col: str,
    held_out_value: object,
) -> tuple[pd.Index, pd.Index]:
    """Hold out an entire operational scenario, policy, design, or source group."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    test_index = df.index[df[group_col] == held_out_value]
    train_index = df.index[df[group_col] != held_out_value]
    if train_index.empty or test_index.empty:
        raise ValueError("Leave-one-group-out split must have non-empty train and test sets")
    return train_index, test_index
