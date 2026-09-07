from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_healthcare_instance
from src.ie.healthcare_flow import (
    build_resource_pressure_projection,
    expected_resource_days_by_cohort,
    expected_transient_visits,
)
from src.markov.transition_utils import load_transition_matrices


def test_fundamental_matrix_expected_visits_match_direct_series() -> None:
    instance = load_healthcare_instance("data")
    matrix = load_transition_matrices(instance.transitions)["c1"]
    visits = expected_transient_visits(matrix)

    # Independent reference: sum Q^k for a long horizon.
    q = matrix[:3, :3]
    series = np.zeros_like(q)
    power = np.eye(3)
    for _ in range(500):
        series += power
        power = power @ q
    assert visits["ED"] == pytest.approx(series[0, 0], rel=1e-9)
    assert visits["ICU"] == pytest.approx(series[0, 1], rel=1e-9)
    assert visits["Ward"] == pytest.approx(series[0, 2], rel=1e-9)


def test_expected_resource_days_are_positive_and_cohort_complete() -> None:
    instance = load_healthcare_instance("data")
    resource_days = expected_resource_days_by_cohort(instance)
    assert len(resource_days) == 4 * 2
    assert set(resource_days["resource"]) == {"ICU", "Ward"}
    assert set(resource_days["cohort"]) == {"c1", "c2", "c3", "c4"}
    assert (resource_days["expected_resource_days_per_arrival"] > 0).all()


def test_littles_law_resource_projection_closes_exactly() -> None:
    instance = load_healthcare_instance("data")
    resource_days = expected_resource_days_by_cohort(instance)
    forecast = pd.DataFrame(
        [
            {"hospital_id": hospital, "cohort": cohort, "predicted_p50": 1.0}
            for hospital in ["H1", "H2", "H3"]
            for cohort in ["c1", "c2", "c3", "c4"]
        ]
    )
    projection = build_resource_pressure_projection(
        instance,
        forecast,
        quantile_column="predicted_p50",
    )

    expected_icu = float(
        resource_days.loc[
            resource_days["resource"] == "ICU", "expected_resource_days_per_arrival"
        ].sum()
    )
    expected_ward = float(
        resource_days.loc[
            resource_days["resource"] == "Ward", "expected_resource_days_per_arrival"
        ].sum()
    )
    for hospital in ["H1", "H2", "H3"]:
        rows = projection[projection["hospital_id"] == hospital].set_index("resource")
        assert rows.loc["ICU", "expected_census"] == pytest.approx(expected_icu)
        assert rows.loc["Ward", "expected_census"] == pytest.approx(expected_ward)
        assert rows.loc["ICU", "projected_utilization"] == pytest.approx(
            rows.loc["ICU", "expected_census"] / rows.loc["ICU", "base_capacity"]
        )
        assert rows.loc["Ward", "safe_capacity"] == pytest.approx(
            rows.loc["Ward", "base_capacity"] * rows.loc["Ward", "safe_utilization"]
        )
