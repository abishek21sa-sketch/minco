from __future__ import annotations

from src.config.loader import load_healthcare_instance
from src.data_engineering.synthetic_arrival_history import generate_synthetic_arrival_history


def test_synthetic_arrival_history_is_seeded_complete_and_nonnegative() -> None:
    instance = load_healthcare_instance("data")
    first = generate_synthetic_arrival_history(instance, n_days=90, seed=12345)
    second = generate_synthetic_arrival_history(instance, n_days=90, seed=12345)

    assert first.equals(second)
    assert len(first) == 90 * 3 * 4
    assert first["date"].nunique() == 90
    assert first["hospital_id"].nunique() == 3
    assert first["cohort"].nunique() == 4
    assert (first["arrivals"] >= 0).all()
    assert first["is_synthetic"].all()
    assert set(first["source_mode"]) == {"synthetic_historical_replay"}
