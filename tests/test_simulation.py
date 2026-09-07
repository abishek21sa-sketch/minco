from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.loader import load_healthcare_instance
from src.simulation.twin_engine import simulate_stochastic_twin


def test_seeded_simulation_is_reproducible(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    left = simulate_stochastic_twin(instance, rng=np.random.default_rng(42))
    right = simulate_stochastic_twin(instance, rng=np.random.default_rng(42))
    pd.testing.assert_frame_equal(left["realized_arrivals"], right["realized_arrivals"])
    pd.testing.assert_frame_equal(left["state_trajectories"], right["state_trajectories"])
