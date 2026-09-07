from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.loader import load_healthcare_instance
from src.markov.forecast_engine import run_forecast
from src.simulation.twin_engine import simulate_stochastic_twin
from src.validation.simulation_validation import (
    _expectation_convergence_table,
    validate_patient_mass_conservation,
)


def test_simulation_conserves_patient_mass(project_root) -> None:
    instance = load_healthcare_instance(project_root / "data")
    outputs = simulate_stochastic_twin(instance, rng=np.random.default_rng(71))
    metrics = validate_patient_mass_conservation(
        outputs["state_trajectories"], outputs["realized_arrivals"]
    )
    assert metrics["mass_conservation_max_error"] == 0.0


def test_monte_carlo_mean_tracks_markov_expectation(project_root) -> None:
    instance = load_healthcare_instance(project_root / "data")
    expected = run_forecast(instance)["state_trajectories"]
    frames = []
    for replication in range(50):
        frame = simulate_stochastic_twin(instance, rng=np.random.default_rng(8000 + replication))[
            "state_trajectories"
        ].copy()
        frame["replication"] = replication
        frames.append(frame)
    table = _expectation_convergence_table(expected, pd.concat(frames, ignore_index=True))
    assert table["absolute_error"].mean() < 0.50
    assert table["normalized_absolute_error"].mean() < 0.10
