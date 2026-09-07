"""AI-parameterized stochastic scenario generation for the capacity MILP.

The scenario generator is the bridge between probabilistic AI and Operations
Research. A fitted Poisson HMM supplies hidden demand-regime dynamics, a
calibrated escalation model supplies ICU-entry probability, and a discharge
hazard model supplies bed-release probability. The resulting scenario tensor
is consumed directly by the stochastic capacity MILP.
"""
from __future__ import annotations

import numpy as np

from .mmpp import PoissonHMMFit, posterior_regime_probabilities


def generate_icu_census_scenarios(
    *,
    hmm_fit: PoissonHMMFit,
    recent_arrival_counts: np.ndarray,
    hospital_arrival_shares: np.ndarray,
    opening_icu_census: np.ndarray,
    discharge_probability_per_period: np.ndarray,
    escalation_probability: np.ndarray,
    n_periods: int,
    n_scenarios: int,
    period_hours: float = 6.0,
    base_observation_hours: float = 1.0,
    seed: int = 20260817,
) -> np.ndarray:
    """Generate ICU census scenarios with shape [scenario,hospital,period]."""
    shares = np.asarray(hospital_arrival_shares, dtype=float)
    opening = np.asarray(opening_icu_census, dtype=float)
    release_p = np.asarray(discharge_probability_per_period, dtype=float)
    escalation_p = np.asarray(escalation_probability, dtype=float)
    h = shares.size
    if not (opening.size == release_p.size == escalation_p.size == h):
        raise ValueError("All hospital vectors must have the same length")
    if h == 0 or n_periods <= 0 or n_scenarios <= 0 or period_hours <= 0 or base_observation_hours <= 0:
        raise ValueError("Invalid scenario dimensions")
    if np.any(shares < 0) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("hospital_arrival_shares must be nonnegative and sum to one")
    if np.any(opening < 0) or np.any((release_p < 0) | (release_p > 1)) or np.any((escalation_p < 0) | (escalation_p > 1)):
        raise ValueError("Invalid census/probability inputs")

    posterior = posterior_regime_probabilities(recent_arrival_counts, hmm_fit)
    current_regime_p = posterior[-1]
    rng = np.random.default_rng(seed)
    tensor = np.zeros((n_scenarios, h, n_periods), dtype=float)
    rate_scale = period_hours / base_observation_hours

    for w in range(n_scenarios):
        regime = int(rng.choice(hmm_fit.n_states, p=current_regime_p))
        census = np.rint(opening).astype(int)
        for t in range(n_periods):
            if t > 0:
                regime = int(rng.choice(hmm_fit.n_states, p=hmm_fit.transition_matrix[regime]))
            total_arrivals = int(rng.poisson(hmm_fit.rates[regime] * rate_scale))
            hospital_arrivals = rng.multinomial(total_arrivals, shares)
            escalations = np.array([
                rng.binomial(int(hospital_arrivals[i]), escalation_p[i]) for i in range(h)
            ], dtype=int)
            releases = np.array([
                rng.binomial(int(max(census[i], 0)), release_p[i]) for i in range(h)
            ], dtype=int)
            census = np.maximum(0, census + escalations - releases)
            tensor[w, :, t] = census
    return tensor
