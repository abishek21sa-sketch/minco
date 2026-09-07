from src.des.patient_flow import DesConfig, DesIntervention, simulate_patient_flow


def test_des_conserves_patients_and_capacity_intervention_improves_ed_wait():
    cfg = DesConfig(horizon_hours=48, arrival_rate_per_hour=4.2, ed_servers=4, seed=91)
    base = simulate_patient_flow(cfg)
    better = simulate_patient_flow(
        cfg,
        DesIntervention(
            extra_ed_servers=2, extra_ward_beds=8, extra_icu_beds=3, transfer_out_capacity_per_6h=2
        ),
    )
    assert base.patient_conservation_error == 0
    assert better.patient_conservation_error == 0
    assert base.max_ward_occupancy <= cfg.ward_beds
    assert base.max_icu_occupancy <= cfg.icu_beds
    assert better.mean_ed_wait_hours <= base.mean_ed_wait_hours
    assert better.evidence_label.startswith("SIMULATED")
