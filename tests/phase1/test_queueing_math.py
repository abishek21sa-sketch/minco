import math

import numpy as np

from src.queueing.hospital_queueing import kingman_gi_g_1_wait, little_law, mm_s_metrics


def test_mm1_reference_case_matches_closed_form_and_little_law():
    lam, mu = 2.0, 3.0
    metrics = mm_s_metrics(lam, mu, 1)
    expected_w = 1.0 / (mu - lam)
    expected_wq = lam / (mu * (mu - lam))
    assert np.isclose(metrics.expected_system_time, expected_w)
    assert np.isclose(metrics.expected_wait, expected_wq)
    assert np.isclose(metrics.expected_system_size, little_law(lam, expected_w))
    assert np.isclose(metrics.expected_queue_length, little_law(lam, expected_wq))


def test_multiserver_queue_reports_instability_and_kingman_variability_effect():
    stable = mm_s_metrics(8.0, 3.0, 4)
    unstable = mm_s_metrics(13.0, 3.0, 4)
    assert stable.stable and stable.probability_wait < 1
    assert not unstable.stable and math.isinf(unstable.expected_wait)
    low_var = kingman_gi_g_1_wait(0.7, 1.0, arrival_scv=0.5, service_scv=0.5)
    high_var = kingman_gi_g_1_wait(0.7, 1.0, arrival_scv=2.0, service_scv=2.0)
    assert high_var > low_var
