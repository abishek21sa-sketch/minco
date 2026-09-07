import numpy as np

from src.minco4x.signature_algorithm import build_markov_flow_forecast, solve_flow_cvar_network


def test_markov_patient_flow_is_row_stochastic_and_explicit():
    f = build_markov_flow_forecast()
    assert np.allclose(f["row_sums"], 1.0)
    assert set(f["forecast_hospital_census"]) == {"H1", "H2"}


def test_network_reference_uses_transfer_or_diversion_recourse():
    r = solve_flow_cvar_network()
    assert r["decision"]["status"] == "OPTIMAL"
    assert r["raw_transfer_total"] > 0
    assert r["raw_deferral_total"] > 0


def test_network_flow_cvar_improves_tail_over_no_cvar():
    r = solve_flow_cvar_network()
    assert r["decision"]["cvar_loss"] <= r["no_cvar_baseline"]["cvar_loss"] + 1e-9
