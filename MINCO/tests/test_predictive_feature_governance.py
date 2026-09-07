from __future__ import annotations

import pandas as pd

from src.validation.predictive_model_governance import (
    TASKS,
    audit_task_features,
    governed_feature_columns,
)


def test_governed_features_exclude_all_future_outcomes() -> None:
    df = pd.DataFrame(
        {
            "time_index": [0, 1],
            "current_value": [1.0, 2.0],
            "replication": [1, 1],
            "target_next_total_unsafe_excess": [2.0, 3.0],
            "target_next_total_blocked_arrivals": [0.0, 1.0],
            "target_next_utilization_critical_flag": [0, 1],
            "next_regime_label": ["surge", "crisis"],
        }
    )
    features = governed_feature_columns(df, "target_next_total_unsafe_excess")
    assert "current_value" in features
    assert "replication" not in features
    assert not any(column.startswith("target_next_") for column in features)
    assert "next_regime_label" not in features


def test_regime_task_detects_deterministic_target_construction() -> None:
    spec = next(item for item in TASKS if item.task_name == "next_regime")
    df = pd.DataFrame(
        {
            "current_regime": ["normal", "normal", "crisis", "surge"],
            "current_regime_code": [0, 0, 2, 1],
            "next_regime_label": ["surge", "surge", "crisis", "crisis"],
            "time_index": [0, 1, 2, 3],
        }
    )
    _, audit = audit_task_features(df, spec)
    flagged = {
        row["feature"]
        for row in audit
        if row["audit_reason"] == "deterministic_target_construction_dependency"
    }
    assert flagged == {"current_regime", "current_regime_code"}
