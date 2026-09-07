from __future__ import annotations

import pandas as pd

from src.validation.predictive_model_governance import TASKS, _task_governance


def test_regression_candidate_is_approved_when_it_beats_baseline_across_partitions() -> None:
    spec = next(item for item in TASKS if item.task_name == "next_unsafe_excess")
    results = pd.DataFrame(
        [
            {
                "task_name": spec.task_name,
                "model_role": "candidate",
                "model_name": "ridge",
                "split_strategy": "chronological_holdout",
                "held_out_group": None,
                "improvement_over_baseline": 0.10,
                "r2": 0.4,
            },
            {
                "task_name": spec.task_name,
                "model_role": "candidate",
                "model_name": "ridge",
                "split_strategy": "grouped_chronological_holdout",
                "held_out_group": None,
                "improvement_over_baseline": 0.08,
                "r2": 0.3,
            },
            {
                "task_name": spec.task_name,
                "model_role": "candidate",
                "model_name": "ridge",
                "split_strategy": "leave_one_scenario_out",
                "held_out_group": "baseline",
                "improvement_over_baseline": 0.05,
                "r2": 0.2,
            },
        ]
    )
    feature_audit = pd.DataFrame(columns=["task_name", "audit_reason", "feature"])
    decision = _task_governance(results, feature_audit, spec)
    assert decision["status"] == "approved_for_reference_case"
    assert decision["selected_model"] == "ridge"
