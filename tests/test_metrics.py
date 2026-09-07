from __future__ import annotations

import pandas as pd
import pytest


def _row(df: pd.DataFrame, policy: str) -> pd.Series:
    matches = df[df["policy_name"] == policy]
    assert len(matches) == 1
    return matches.iloc[0]


def test_headline_baseline_claims_match_stored_evidence() -> None:
    df = pd.DataFrame(
        {
            "scenario": ["baseline", "baseline"],
            "policy_name": ["no_control", "optimized_network"],
            "total_unsafe_excess_mean": [100.0, 58.103],
            "max_utilization_ratio_mean": [1.0, 0.88937],
        }
    )
    baseline = df[df["scenario"] == "baseline"]
    no_control = _row(baseline, "no_control")
    optimized = _row(baseline, "optimized_network")

    unsafe_reduction = (
        100
        * (no_control["total_unsafe_excess_mean"] - optimized["total_unsafe_excess_mean"])
        / no_control["total_unsafe_excess_mean"]
    )
    utilization_reduction = (
        100
        * (no_control["max_utilization_ratio_mean"] - optimized["max_utilization_ratio_mean"])
        / no_control["max_utilization_ratio_mean"]
    )

    assert unsafe_reduction == pytest.approx(41.897, abs=0.01)
    assert utilization_reduction == pytest.approx(11.063, abs=0.01)
