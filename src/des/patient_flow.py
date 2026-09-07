"""Small, transparent patient-level discrete-event simulator.

This DES intentionally focuses on the operational quantities the workstation
needs: ED waiting, bed boarding, ICU/Ward occupancy, throughput, and diversion.
It is not a clinical disease model. Randomness is seeded and the result is
labeled SIMULATED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DesConfig:
    horizon_hours: float = 72.0
    arrival_rate_per_hour: float = 3.0
    ed_servers: int = 5
    ed_service_mean_hours: float = 1.8
    ward_beds: int = 30
    icu_beds: int = 10
    ward_los_mean_hours: float = 30.0
    icu_los_mean_hours: float = 42.0
    admission_probability: float = 0.34
    icu_given_admission_probability: float = 0.18
    max_boarding_hours: float = 12.0
    seed: int = 20260823

    def validate(self) -> None:
        positive = [self.horizon_hours, self.arrival_rate_per_hour, self.ed_service_mean_hours,
                    self.ward_los_mean_hours, self.icu_los_mean_hours, self.max_boarding_hours]
        if any(v <= 0 for v in positive):
            raise ValueError("DES time/rate parameters must be positive")
        if self.ed_servers < 1 or self.ward_beds < 0 or self.icu_beds < 0:
            raise ValueError("DES capacities are invalid")
        for p in [self.admission_probability, self.icu_given_admission_probability]:
            if not 0 <= p <= 1:
                raise ValueError("DES probabilities must lie in [0,1]")


@dataclass(frozen=True)
class DesIntervention:
    extra_ward_beds: int = 0
    extra_icu_beds: int = 0
    extra_ed_servers: int = 0
    admission_probability_multiplier: float = 1.0
    transfer_out_capacity_per_6h: int = 0


@dataclass(frozen=True)
class DesResult:
    arrivals: int
    completed_ed: int
    admitted_ward: int
    admitted_icu: int
    discharged: int
    transferred_out: int
    diverted: int
    mean_ed_wait_hours: float
    p95_ed_wait_hours: float
    mean_boarding_hours: float
    p95_boarding_hours: float
    max_ward_occupancy: int
    max_icu_occupancy: int
    final_ward_occupancy: int
    final_icu_occupancy: int
    patient_conservation_error: int
    evidence_label: str = "SIMULATED_DISCRETE_EVENT_RESULT"


@dataclass(order=True)
class _Event:
    time: float
    sequence: int
    kind: str = field(compare=False)
    patient_id: int = field(compare=False)
    payload: dict = field(default_factory=dict, compare=False)


def _exp(rng: np.random.Generator, mean: float) -> float:
    return float(rng.exponential(mean))


def simulate_patient_flow(config: DesConfig, intervention: DesIntervention | None = None) -> DesResult:
    config.validate()
    intervention = intervention or DesIntervention()
    rng = np.random.default_rng(config.seed)
    ed_servers = config.ed_servers + max(0, intervention.extra_ed_servers)
    ward_capacity = config.ward_beds + max(0, intervention.extra_ward_beds)
    icu_capacity = config.icu_beds + max(0, intervention.extra_icu_beds)
    admission_p = min(1.0, max(0.0, config.admission_probability * intervention.admission_probability_multiplier))

    queue: list[_Event] = []
    sequence = 0
    next_patient = 0
    now = 0.0
    ed_busy = 0
    ward_occ = 0
    icu_occ = 0
    ward_waiting: list[tuple[int, float]] = []
    icu_waiting: list[tuple[int, float]] = []
    ed_waiting: list[tuple[int, float]] = []
    transfer_tokens = intervention.transfer_out_capacity_per_6h
    next_transfer_refresh = 6.0

    arrivals = completed_ed = admitted_ward = admitted_icu = discharged = transferred = diverted = 0
    ed_waits: list[float] = []
    boarding_waits: list[float] = []
    max_ward = max_icu = 0

    def push(time: float, kind: str, patient_id: int, **payload) -> None:
        nonlocal sequence
        sequence += 1
        heapq.heappush(queue, _Event(time, sequence, kind, patient_id, payload))

    def start_ed(pid: int, arrival_time: float, time: float) -> None:
        nonlocal ed_busy
        ed_busy += 1
        ed_waits.append(max(0.0, time - arrival_time))
        push(time + _exp(rng, config.ed_service_mean_hours), "ed_complete", pid)

    def try_admit_waiters(time: float) -> None:
        nonlocal ward_occ, icu_occ, admitted_ward, admitted_icu, transferred, transfer_tokens
        changed = True
        while changed:
            changed = False
            if icu_waiting and icu_occ < icu_capacity:
                pid, request_time = icu_waiting.pop(0)
                icu_occ += 1; admitted_icu += 1; changed = True
                boarding_waits.append(time - request_time)
                push(time + _exp(rng, config.icu_los_mean_hours), "icu_discharge", pid)
            elif icu_waiting and transfer_tokens > 0 and (time - icu_waiting[0][1]) >= config.max_boarding_hours:
                pid, request_time = icu_waiting.pop(0)
                transfer_tokens -= 1; transferred += 1; changed = True
                boarding_waits.append(time - request_time)
            if ward_waiting and ward_occ < ward_capacity:
                pid, request_time = ward_waiting.pop(0)
                ward_occ += 1; admitted_ward += 1; changed = True
                boarding_waits.append(time - request_time)
                push(time + _exp(rng, config.ward_los_mean_hours), "ward_discharge", pid)
            elif ward_waiting and transfer_tokens > 0 and (time - ward_waiting[0][1]) >= config.max_boarding_hours:
                pid, request_time = ward_waiting.pop(0)
                transfer_tokens -= 1; transferred += 1; changed = True
                boarding_waits.append(time - request_time)

    # Seed all arrivals using a Poisson process, including enough beyond horizon to stop cleanly.
    t = _exp(rng, 1.0 / config.arrival_rate_per_hour)
    while t <= config.horizon_hours:
        next_patient += 1
        push(t, "arrival", next_patient)
        t += _exp(rng, 1.0 / config.arrival_rate_per_hour)

    while queue:
        event = heapq.heappop(queue)
        now = event.time
        if now > config.horizon_hours:
            break
        while now >= next_transfer_refresh:
            transfer_tokens = intervention.transfer_out_capacity_per_6h
            next_transfer_refresh += 6.0
        kind, pid = event.kind, event.patient_id
        if kind == "arrival":
            arrivals += 1
            if ed_busy < ed_servers:
                start_ed(pid, now, now)
            else:
                ed_waiting.append((pid, now))
        elif kind == "ed_complete":
            completed_ed += 1
            ed_busy -= 1
            if ed_waiting:
                waiting_pid, arrival_time = ed_waiting.pop(0)
                start_ed(waiting_pid, arrival_time, now)
            if rng.random() < admission_p:
                if rng.random() < config.icu_given_admission_probability:
                    if icu_occ < icu_capacity:
                        icu_occ += 1; admitted_icu += 1
                        boarding_waits.append(0.0)
                        push(now + _exp(rng, config.icu_los_mean_hours), "icu_discharge", pid)
                    else:
                        icu_waiting.append((pid, now))
                else:
                    if ward_occ < ward_capacity:
                        ward_occ += 1; admitted_ward += 1
                        boarding_waits.append(0.0)
                        push(now + _exp(rng, config.ward_los_mean_hours), "ward_discharge", pid)
                    else:
                        ward_waiting.append((pid, now))
            else:
                discharged += 1
        elif kind == "ward_discharge":
            ward_occ = max(0, ward_occ - 1); discharged += 1
            try_admit_waiters(now)
        elif kind == "icu_discharge":
            icu_occ = max(0, icu_occ - 1); discharged += 1
            try_admit_waiters(now)
        max_ward = max(max_ward, ward_occ)
        max_icu = max(max_icu, icu_occ)

    # Any patient still waiting in ED or for a bed is operationally unresolved at horizon, not disappeared.
    unresolved = len(ed_waiting) + len(ward_waiting) + len(icu_waiting) + ed_busy + ward_occ + icu_occ
    accounted = discharged + transferred + unresolved
    conservation_error = arrivals - accounted
    # A completion outside the horizon may leave an arrival busy in ED; included in unresolved above.

    def avg(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0
    def p95(xs: list[float]) -> float:
        return float(np.quantile(xs, 0.95)) if xs else 0.0

    return DesResult(
        arrivals=arrivals,
        completed_ed=completed_ed,
        admitted_ward=admitted_ward,
        admitted_icu=admitted_icu,
        discharged=discharged,
        transferred_out=transferred,
        diverted=diverted,
        mean_ed_wait_hours=avg(ed_waits),
        p95_ed_wait_hours=p95(ed_waits),
        mean_boarding_hours=avg(boarding_waits),
        p95_boarding_hours=p95(boarding_waits),
        max_ward_occupancy=max_ward,
        max_icu_occupancy=max_icu,
        final_ward_occupancy=ward_occ,
        final_icu_occupancy=icu_occ,
        patient_conservation_error=conservation_error,
    )
