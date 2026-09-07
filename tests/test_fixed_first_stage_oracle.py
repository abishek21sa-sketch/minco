import numpy as np
import pytest

from src.decision_math.stochastic_milp_oracle import solve_small_stochastic_milp
from src.minco4x.signature_algorithm import make_network_flow_instance


def test_fixed_first_stage_is_respected():
    inst, _ = make_network_flow_instance()
    z = np.zeros((2, 1), dtype=int)
    s = solve_small_stochastic_milp(inst, fixed_surge=z, fixed_flex=z)
    assert s.status == "OPTIMAL" and int(s.surge.sum()) == 0 and int(s.flex_blocks.sum()) == 0


def test_fixed_first_stage_shape_is_validated():
    inst, _ = make_network_flow_instance()
    with pytest.raises(ValueError):
        solve_small_stochastic_milp(inst, fixed_surge=np.zeros((1, 1)))
    with pytest.raises(ValueError):
        solve_small_stochastic_milp(inst, fixed_flex=np.zeros((1, 1)))
