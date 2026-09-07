import numpy as np

from src.stochastic.decision_scenarios import generate_icu_census_scenarios
from src.stochastic.mmpp import fit_poisson_hmm, simulate_mmpp


def test_ai_parameterized_scenarios_are_reproducible_and_shape_correct():
    a = np.array([[0.93, 0.07], [0.10, 0.90]])
    rates = np.array([2.0, 6.0])
    _, counts = simulate_mmpp(transition_matrix=a, rates=rates, n_periods=500, seed=5)
    fit = fit_poisson_hmm(counts, n_states=2, max_iter=60, tolerance=1e-4, random_state=5)
    kwargs = dict(
        hmm_fit=fit,
        recent_arrival_counts=counts[-72:],
        hospital_arrival_shares=np.array([0.40, 0.35, 0.25]),
        opening_icu_census=np.array([20, 16, 13]),
        discharge_probability_per_period=np.array([0.08, 0.09, 0.10]),
        escalation_probability=np.array([0.18, 0.16, 0.14]),
        n_periods=6,
        n_scenarios=50,
        seed=99,
    )
    one = generate_icu_census_scenarios(**kwargs)
    two = generate_icu_census_scenarios(**kwargs)
    assert one.shape == (50, 3, 6)
    assert np.array_equal(one, two)
    assert np.all(one >= 0)
    assert np.var(one[:, :, -1]) > 0
