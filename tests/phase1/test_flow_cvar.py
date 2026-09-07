from dataclasses import replace

import numpy as np

from src.decision_math.flow_cvar import cvar_ablation, solve_flow_cvar
from src.decision_math.stochastic_milp_oracle import make_tiny_oracle_instance


def test_flow_cvar_returns_feasible_auditable_decision():
    decision = solve_flow_cvar(make_tiny_oracle_instance())
    assert decision.algorithm == "FLOW-CVaR"
    assert decision.status == "OPTIMAL"
    assert decision.feasible
    assert decision.cvar_loss >= decision.expected_recourse_loss - 1e-9
    assert decision.surge_activations >= 0
    assert decision.flex_staff_blocks >= 0
    assert "not a clinical outcome guarantee" in decision.bounded_claim


def test_flow_cvar_tail_risk_weight_can_change_capacity_decision():
    # Construct a rare severe surge where expected-cost planning tolerates the tail,
    # but sufficiently risk-averse FLOW-CVaR activates capacity.
    base = make_tiny_oracle_instance()
    inst = replace(
        base,
        scenario_probabilities=np.array([0.9, 0.1]),
        demand=np.array([[[9.0]], [[20.0]]]),
        surge_cost=np.array([18.0]),
        cvar_alpha=0.8,
        cvar_weight=1.5,
    )
    result = cvar_ablation(inst)
    governed, no_cvar = result["flow_cvar"], result["no_cvar"]
    assert governed.feasible and no_cvar.feasible
    assert governed.cvar_loss <= no_cvar.cvar_loss + 1e-9
    assert (governed.surge_activations, governed.flex_staff_blocks) != (
        no_cvar.surge_activations,
        no_cvar.flex_staff_blocks,
    )
