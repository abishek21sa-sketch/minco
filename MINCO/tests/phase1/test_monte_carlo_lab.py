import numpy as np

from src.stochastic.monte_carlo import (
    CapacityPolicy,
    MonteCarloConfig,
    evaluate_policies_common_random_numbers,
    paired_policy_difference,
)


def test_common_random_number_policy_comparison_reuses_scenario_seeds_and_rewards_capacity():
    a = np.array(
        [
            [0.90, 0.08, 0.02],
            [0.10, 0.82, 0.08],
            [0.03, 0.12, 0.85],
        ]
    )
    rates = np.array([1.5, 2.5, 4.0])
    policies = [
        CapacityPolicy("current"),
        CapacityPolicy("surge", icu_surge_beds=10, ward_surge_beds=20, flex_staff_beds=4),
    ]
    detail, summary = evaluate_policies_common_random_numbers(
        policies,
        transition_matrix=a,
        regime_rates=rates,
        config=MonteCarloConfig(horizon_hours=48, n_scenarios=80, seed=12),
    )
    seed_counts = detail.groupby("scenario_id")["scenario_seed"].nunique()
    assert (seed_counts == 1).all()
    assert len(detail) == 160
    paired = paired_policy_difference(detail, baseline_policy="current", candidate_policy="surge")
    assert paired.shape == (80,)
    expected = summary.set_index("policy")["expected_loss"]
    assert expected["surge"] <= expected["current"]
