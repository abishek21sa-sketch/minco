"""Stochastic healthcare demand and patient-flow mathematics."""

from .mmpp import (
    PoissonHMMFit,
    fit_poisson_hmm,
    posterior_regime_probabilities,
    simulate_mmpp,
    stationary_distribution,
)

__all__ = [
    "PoissonHMMFit",
    "fit_poisson_hmm",
    "posterior_regime_probabilities",
    "simulate_mmpp",
    "stationary_distribution",
]
from .ctmc import CTMCModel, make_reference_patient_flow_ctmc, simulate_population_counts

__all__ += ["CTMCModel", "make_reference_patient_flow_ctmc", "simulate_population_counts"]
from .monte_carlo import (
    CapacityPolicy,
    MonteCarloConfig,
    evaluate_policies_common_random_numbers,
    paired_policy_difference,
)

__all__ += [
    "CapacityPolicy",
    "MonteCarloConfig",
    "evaluate_policies_common_random_numbers",
    "paired_policy_difference",
]

from .decision_scenarios import generate_icu_census_scenarios
__all__ += ["generate_icu_census_scenarios"]
