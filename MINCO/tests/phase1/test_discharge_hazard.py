import numpy as np
import pandas as pd

from src.ml.discharge_hazard import DiscreteTimeHazardModel, synthetic_stay_dataset


def test_discrete_time_hazard_returns_monotone_discharge_probabilities():
    stays = synthetic_stay_dataset(900, seed=18)
    model = DiscreteTimeHazardModel(max_horizon_hours=96).fit(stays)
    patients = pd.DataFrame(
        [
            {"age": 45, "acuity_score": 0.2, "unit": "Ward", "cohort": "medical"},
            {"age": 78, "acuity_score": 0.8, "unit": "ICU", "cohort": "respiratory"},
        ]
    )
    curves = model.predict_hazard_curve(patients, [6, 12, 24, 48, 72, 96])
    for _, group in curves.groupby("patient_row"):
        p = group.sort_values("horizon_hours")["discharge_probability"].to_numpy()
        assert np.all(np.diff(p) >= -1e-12)
        assert np.all((p >= 0) & (p <= 1))
    p24 = (
        curves[curves.horizon_hours == 24]
        .sort_values("patient_row")["discharge_probability"]
        .to_numpy()
    )
    assert p24[0] > p24[1]
