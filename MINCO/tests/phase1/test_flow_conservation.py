import numpy as np

from src.decision_math.flow_conservation import (
    close_census,
    maximum_staffed_beds,
    staffing_adequacy_ratio,
    verify_flow_balance,
)


def test_patient_flow_conservation_and_staffed_capacity():
    closing = close_census(
        opening_census=25, arrivals=8, transfers_in=2, discharges=7, transfers_out=3
    )
    assert closing == 25
    balance = verify_flow_balance(
        opening_census=25,
        arrivals=8,
        transfers_in=2,
        discharges=7,
        transfers_out=3,
        observed_closing_census=25,
    )
    assert np.isclose(balance.residual, 0.0)
    assert (
        maximum_staffed_beds(available_nurse_hours=120, nurse_hours_per_bed=6, physical_beds=30)
        == 20
    )
    assert staffing_adequacy_ratio(available_nurse_hours=90, required_nurse_hours=100) == 0.9
