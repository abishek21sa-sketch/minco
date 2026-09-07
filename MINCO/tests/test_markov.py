from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.loader import load_healthcare_instance
from src.markov.forecast_engine import run_forecast
from src.markov.transition_utils import load_transition_matrices


def test_transition_matrices_are_row_stochastic(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    matrices = load_transition_matrices(instance.transitions)
    assert matrices
    for matrix in matrices.values():
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(matrix.shape[0]), atol=1e-10)


def test_forecast_is_deterministic(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    left = run_forecast(instance)
    right = run_forecast(instance)
    pd.testing.assert_frame_equal(left["state_trajectories"], right["state_trajectories"])


def test_forecast_conserves_expected_patient_mass(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    output = run_forecast(instance)["state_trajectories"]
    observed = output.groupby(["day", "hospital_id", "cohort"])["expected_count"].sum()
    arrivals = instance.arrivals.df.copy()
    arrivals["expected_total"] = arrivals.groupby(["hospital_id", "cohort"])["arrivals"].cumsum()
    expected = arrivals.set_index(["day", "hospital_id", "cohort"])["expected_total"].sort_index()
    np.testing.assert_allclose(observed.sort_index().to_numpy(), expected.to_numpy(), atol=1e-8)
