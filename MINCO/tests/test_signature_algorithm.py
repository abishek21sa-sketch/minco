import numpy as np

from src.minco4x.signature_algorithm import build_markov_flow_forecast, solve_flow_cvar_network


def test_signature_markov_network_and_tail_risk():
    f = build_markov_flow_forecast()
    assert np.allclose(f["row_sums"], 1)
    assert len(f["forecast_hospital_census"]) == 2
    r = solve_flow_cvar_network()
    assert r["decision"]["status"] == "OPTIMAL"
    assert r["decision"]["cvar_loss"] <= r["no_cvar_baseline"]["cvar_loss"] + 1e-9
    assert r["raw_transfer_total"] > 0
