from __future__ import annotations

import pandas as pd

from src.contracts.metrics import OperationalMetrics
from src.optimization.action_summary import summarize_policy_actions


def test_policy_action_summary_aggregates_solved_decisions() -> None:
    snapshot = {
        "surge": pd.DataFrame({"surge_activated": [2.0, 3.5]}),
        "elective_rejected": pd.DataFrame({"elective_rejected": [1.0, 0.5]}),
        "icu_transfers": pd.DataFrame({"icu_transfer_load": [4.0, 2.0]}),
        "model_metrics": pd.DataFrame(
            [
                {
                    "model_status": 2,
                    "objective_value": 123.4,
                    "is_mip": False,
                    "mip_gap": None,
                    "best_bound": None,
                }
            ]
        ),
    }
    summary = summarize_policy_actions(snapshot)
    assert summary["total_surge_activated"] == 5.5
    assert summary["total_elective_rejected"] == 1.5
    assert summary["total_icu_transfer_load"] == 6.0
    assert summary["solver_status_label"] == "OPTIMAL"
    assert summary["objective_value"] == 123.4

    metrics = OperationalMetrics.from_mapping({**summary, "n_replications": 3})
    assert metrics.solver_status_label == "OPTIMAL"
    assert metrics.total_surge_activated == 5.5
