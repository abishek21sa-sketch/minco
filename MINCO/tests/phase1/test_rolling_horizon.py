from src.decision_math.rolling_horizon import rolling_horizon_reference
from src.decision_math.stochastic_milp_oracle import make_tiny_oracle_instance


def test_rolling_horizon_reoptimizes_when_demand_changes():
    def factory(epoch: int):
        inst = make_tiny_oracle_instance()
        if epoch == 1:
            return inst
        return type(inst)(**{**inst.__dict__, "demand": inst.demand + 5.0})

    decisions = rolling_horizon_reference([1, 2], factory)
    assert len(decisions) == 2
    assert all(item.status == "OPTIMAL" for item in decisions)
    # Demand stress should not make the optimized objective smaller.
    assert decisions[1].objective >= decisions[0].objective - 1e-8
