import numpy as np

from src.decision_math.progressive_hedging import progressive_hedging_single_hospital
from src.decision_math.stochastic_milp_oracle import (
    brute_force_single_hospital_first_stage,
    make_tiny_oracle_instance,
    solve_small_stochastic_milp,
)


def test_highs_stochastic_milp_matches_independent_enumeration_oracle():
    inst = make_tiny_oracle_instance()
    solved = solve_small_stochastic_milp(inst)
    enumerated = brute_force_single_hospital_first_stage(inst)
    assert solved.status == "OPTIMAL"
    assert np.isclose(solved.objective, enumerated["objective"], atol=1e-6)
    assert int(solved.surge[0, 0]) == enumerated["surge"]
    assert int(solved.flex_blocks[0, 0]) == enumerated["flex"]
    assert np.all(solved.xi >= -1e-8)


def test_progressive_hedging_reaches_same_first_stage_consensus_on_tiny_instance():
    inst = make_tiny_oracle_instance()
    extensive = solve_small_stochastic_milp(inst)
    ph = progressive_hedging_single_hospital(inst, rho=20.0, max_iter=80)
    assert ph.converged
    assert ph.consensus_surge == int(extensive.surge[0, 0])
    assert ph.consensus_flex == int(extensive.flex_blocks[0, 0])
    assert ph.residual_history[-1] <= ph.residual_history[0] + 1e-9
