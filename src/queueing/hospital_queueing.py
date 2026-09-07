"""Queueing-theory reference calculations for hospital service stations.

These formulas are used as analytical checks against stochastic simulation,
not as claims that every hospital process is exactly Markovian.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial


@dataclass(frozen=True)
class QueueMetrics:
    arrival_rate: float
    service_rate_per_server: float
    servers: int
    utilization: float
    probability_wait: float
    expected_wait: float
    expected_system_time: float
    expected_queue_length: float
    expected_system_size: float
    stable: bool
    time_unit: str = "hours"


def little_law(arrival_rate: float, mean_time: float) -> float:
    if arrival_rate < 0 or mean_time < 0:
        raise ValueError("arrival_rate and mean_time must be nonnegative")
    return float(arrival_rate * mean_time)


def erlang_c_probability_wait(arrival_rate: float, service_rate: float, servers: int) -> float:
    """Erlang-C probability that an arrival waits in an M/M/s queue."""
    if arrival_rate < 0 or service_rate <= 0 or servers <= 0:
        raise ValueError("Invalid queue parameters")
    if arrival_rate == 0:
        return 0.0
    offered_load = arrival_rate / service_rate
    rho = offered_load / servers
    if rho >= 1.0:
        return 1.0
    base = sum(offered_load**k / factorial(k) for k in range(servers))
    tail = offered_load**servers / (factorial(servers) * (1.0 - rho))
    return float(tail / (base + tail))


def mm_s_metrics(arrival_rate: float, service_rate: float, servers: int) -> QueueMetrics:
    if arrival_rate < 0 or service_rate <= 0 or servers <= 0:
        raise ValueError("Invalid queue parameters")
    rho = arrival_rate / (servers * service_rate)
    if arrival_rate == 0:
        return QueueMetrics(
            arrival_rate, service_rate, servers, 0.0, 0.0, 0.0,
            1.0 / service_rate, 0.0, 0.0, True,
        )
    if rho >= 1.0:
        return QueueMetrics(
            arrival_rate, service_rate, servers, rho, 1.0,
            float("inf"), float("inf"), float("inf"), float("inf"), False,
        )
    p_wait = erlang_c_probability_wait(arrival_rate, service_rate, servers)
    wq = p_wait / (servers * service_rate - arrival_rate)
    w = wq + 1.0 / service_rate
    lq = little_law(arrival_rate, wq)
    l = little_law(arrival_rate, w)
    return QueueMetrics(
        arrival_rate, service_rate, servers, float(rho), float(p_wait),
        float(wq), float(w), float(lq), float(l), True,
    )


def kingman_gi_g_1_wait(
    arrival_rate: float,
    service_rate: float,
    *,
    arrival_scv: float,
    service_scv: float,
) -> float:
    """Kingman's approximation for GI/G/1 expected queue wait.

    SCV denotes squared coefficient of variation. Returns infinity when the
    station is unstable.
    """
    if arrival_rate < 0 or service_rate <= 0 or arrival_scv < 0 or service_scv < 0:
        raise ValueError("Invalid queue parameters")
    rho = arrival_rate / service_rate
    if rho >= 1.0:
        return float("inf")
    if arrival_rate == 0:
        return 0.0
    variability = 0.5 * (arrival_scv + service_scv)
    mean_service = 1.0 / service_rate
    return float((rho / (1.0 - rho)) * variability * mean_service)


def staffed_bed_utilization(census: float, staffed_beds: float) -> float:
    if census < 0 or staffed_beds <= 0:
        raise ValueError("census must be nonnegative and staffed_beds positive")
    return float(census / staffed_beds)
