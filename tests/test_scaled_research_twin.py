from pathlib import Path

import numpy as np

from src.benchmarks.stochastic_capacity import make_network_capacity_benchmark
from src.data_runtime.memory_store import MemoryEventStore
from src.runtime.workstation_service import WorkstationService
from src.scenarios.operating_modes import get_operating_mode
from src.scenarios.scaled_replay import build_scaled_event_replay


def test_scaled_replay_is_100k_plus_multi_facility_and_deterministic():
    kwargs = dict(
        n_hospitals=120,
        horizon_days=30,
        minimum_events=100_000,
        seed=20260906,
        public_reference_csv=Path("data/public_reference/cms_hospital_general_information.csv"),
    )
    first = build_scaled_event_replay(**kwargs)
    second = build_scaled_event_replay(**kwargs)
    assert len(first) >= 100_000
    assert len({event.hospital_id for event in first}) == 120
    assert first[0].event_id == second[0].event_id
    assert first[-1].event_id == second[-1].event_id
    assert all(event.source_mode.value == "synthetic" for event in first)
    capacity_events = [event for event in first if event.event_type.value == "capacity_change"]
    patient_events = [event for event in first if event.patient_id]
    assert capacity_events and all(
        event.metadata["capacity_claim"] == "synthetic_calibration" for event in capacity_events
    )
    assert patient_events and all(
        event.metadata["clinical_use"] == "not_validated_for_patient_care"
        for event in patient_events
    )


def test_operating_modes_change_model_inputs_without_changing_contract():
    normal = get_operating_mode("normal")
    covid = get_operating_mode("covid_like")
    assert covid.demand_multiplier > normal.demand_multiplier
    normal_instance = make_network_capacity_benchmark(
        n_hospitals=12, n_periods=3, n_scenarios=4, operating_mode="normal"
    )
    covid_instance = make_network_capacity_benchmark(
        n_hospitals=12, n_periods=3, n_scenarios=4, operating_mode="covid_like"
    )
    assert covid_instance.demand.shape == normal_instance.demand.shape
    assert float(np.mean(covid_instance.demand)) > float(np.mean(normal_instance.demand))


def test_runtime_exposes_research_scale_catalog_for_legacy_memory_adapter():
    service = WorkstationService(store=MemoryEventStore())
    status = service.status()
    catalog = service.scenario_catalog()
    assert status.facility_count == 3
    assert len(catalog["operating_modes"]) >= 5
    assert catalog["claim_boundary"].startswith("Modes are synthetic")
