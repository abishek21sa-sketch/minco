from src.decision_math.stochastic_milp_oracle import make_tiny_oracle_instance
from src.optimization_julia.bridge import instance_to_payload


def test_python_to_julia_stochastic_instance_payload_preserves_dimensions_and_cvar():
    inst = make_tiny_oracle_instance()
    payload = instance_to_payload(inst)
    assert payload["hospitals"] == ["H1"]
    assert len(payload["demand"]) == 2
    assert len(payload["demand"][0]) == 1
    assert payload["cvar_alpha"] == 0.8
    assert payload["cvar_weight"] == 0.4
    assert payload["transfer_capacity"] == [[[0.0]]]
