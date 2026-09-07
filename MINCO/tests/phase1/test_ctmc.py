import numpy as np

from src.stochastic.ctmc import (
    CTMCModel,
    make_reference_patient_flow_ctmc,
    simulate_population_counts,
)


def test_ctmc_transition_matrix_and_absorption_math():
    model = make_reference_patient_flow_ctmc()
    p = model.transition_matrix(6.0)
    assert np.all(p >= -1e-12)
    assert np.allclose(p.sum(axis=1), 1.0)
    times = model.expected_transient_time("ED")
    assert times["ED"] > 0
    assert times["ICU"] > 0
    assert model.expected_time_to_absorption("ED") == sum(times.values())
    absorption = model.absorption_probabilities("ED")
    assert np.isclose(sum(absorption.values()), 1.0)
    assert 0 < absorption["Transferred"] < 1


def test_ctmc_known_two_state_expected_absorption_time():
    rate = 0.25
    q = np.array([[-rate, rate], [0.0, 0.0]])
    model = CTMCModel(states=("Care", "Discharged"), generator=q, absorbing_states=("Discharged",))
    assert np.isclose(model.expected_time_to_absorption("Care"), 1.0 / rate)


def test_ctmc_population_simulation_conserves_patients_plus_arrivals():
    model = make_reference_patient_flow_ctmc()
    arrivals = [3, 4, 2, 5, 1, 0, 3]
    trajectory = simulate_population_counts(
        model, initial_counts=[2, 1, 0, 0, 0, 0], arrivals_by_step=arrivals, seed=44
    )
    expected_totals = 3 + np.cumsum([0] + arrivals)
    assert np.array_equal(trajectory.sum(axis=1), expected_totals)
