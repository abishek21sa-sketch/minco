from __future__ import annotations

import pandas as pd

from src.validation.claim_governance import paired_policy_comparison


def test_paired_claim_governance_distinguishes_supported_advantage() -> None:
    rows = []
    for replication in range(1, 41):
        rows.append(
            {"scenario": "x", "replication": replication, "policy_name": "a", "metric": 5.0}
        )
        rows.append(
            {"scenario": "x", "replication": replication, "policy_name": "b", "metric": 8.0}
        )
    result = paired_policy_comparison(
        pd.DataFrame(rows),
        policy_a="a",
        policy_b="b",
        metric="metric",
        bootstrap_iterations=500,
    )
    assert result.iloc[0]["claim_classification"] == "validated_advantage"
    assert result.iloc[0]["bootstrap_ci95_high"] < 0
