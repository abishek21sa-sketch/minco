from __future__ import annotations

import pandas as pd

from src.validation.temporal_splits import (
    chronological_holdout,
    grouped_chronological_holdout,
    leave_one_group_out,
)


def _frame() -> pd.DataFrame:
    rows = []
    for scenario in ["baseline", "surge"]:
        for replication in [0, 1]:
            for time_index in range(8):
                rows.append(
                    {
                        "scenario": scenario,
                        "replication": replication,
                        "time_index": time_index,
                    }
                )
    return pd.DataFrame(rows)


def test_global_chronological_holdout_has_no_future_in_training() -> None:
    df = _frame()
    train_idx, test_idx, metadata = chronological_holdout(df, test_fraction=0.25)
    assert set(train_idx).isdisjoint(set(test_idx))
    assert metadata.train_time_max < metadata.test_time_min
    assert sorted(df.loc[test_idx, "time_index"].unique().tolist()) == [6, 7]


def test_grouped_chronological_holdout_preserves_each_trajectory() -> None:
    df = _frame()
    train_idx, test_idx, metadata = grouped_chronological_holdout(
        df,
        group_cols=["scenario", "replication"],
        test_fraction=0.25,
    )
    assert set(train_idx).isdisjoint(set(test_idx))
    assert metadata.group_columns == ("scenario", "replication")
    for _, group in df.groupby(["scenario", "replication"]):
        group_train = group.loc[group.index.intersection(train_idx), "time_index"]
        group_test = group.loc[group.index.intersection(test_idx), "time_index"]
        assert group_train.max() < group_test.min()


def test_leave_one_scenario_out_has_no_scenario_overlap() -> None:
    df = _frame()
    train_idx, test_idx = leave_one_group_out(
        df,
        group_col="scenario",
        held_out_value="surge",
    )
    assert set(df.loc[train_idx, "scenario"]) == {"baseline"}
    assert set(df.loc[test_idx, "scenario"]) == {"surge"}
