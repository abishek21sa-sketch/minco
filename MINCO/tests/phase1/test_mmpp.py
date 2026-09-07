import numpy as np

from src.stochastic.mmpp import (
    fit_poisson_hmm,
    posterior_regime_probabilities,
    simulate_mmpp,
    stationary_distribution,
)


def test_mmpp_simulation_and_stationary_distribution_are_reproducible():
    a = np.array([[0.94, 0.06], [0.12, 0.88]])
    rates = np.array([4.0, 14.0])
    s1, y1 = simulate_mmpp(transition_matrix=a, rates=rates, n_periods=500, seed=17)
    s2, y2 = simulate_mmpp(transition_matrix=a, rates=rates, n_periods=500, seed=17)
    assert np.array_equal(s1, s2)
    assert np.array_equal(y1, y2)
    pi = stationary_distribution(a)
    assert np.all(pi >= 0)
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi @ a, pi)


def test_poisson_hmm_em_recovers_ordered_demand_regimes_and_improves_likelihood():
    a = np.array([[0.965, 0.035], [0.06, 0.94]])
    rates = np.array([5.0, 17.0])
    _, counts = simulate_mmpp(transition_matrix=a, rates=rates, n_periods=1800, seed=2026)
    fit = fit_poisson_hmm(counts, n_states=2, max_iter=120, random_state=9)
    assert fit.rates[0] < fit.rates[1]
    assert abs(fit.rates[0] - 5.0) < 2.0
    assert abs(fit.rates[1] - 17.0) < 3.0
    assert fit.log_likelihood_history[-1] >= fit.log_likelihood_history[0] - 1e-7
    posterior = posterior_regime_probabilities(counts, fit)
    assert posterior.shape == (len(counts), 2)
    assert np.allclose(posterior.sum(axis=1), 1.0)
