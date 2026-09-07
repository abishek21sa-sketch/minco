"""Continuous-time Markov patient-flow model for MINCO.

The state process X(t) follows a continuous-time Markov chain with generator Q:

    P(t) = exp(Q t)

For transient generator T, the continuous-time fundamental matrix is

    N = (-T)^(-1)

and N[i,j] is the expected time spent in transient state j before absorption
when starting in transient state i.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class CTMCModel:
    states: tuple[str, ...]
    generator: np.ndarray
    absorbing_states: tuple[str, ...]
    time_unit: str = "hours"

    def __post_init__(self) -> None:
        q = np.asarray(self.generator, dtype=float)
        if q.shape != (len(self.states), len(self.states)):
            raise ValueError("Generator shape must match state count")
        if len(set(self.states)) != len(self.states):
            raise ValueError("State names must be unique")
        if not set(self.absorbing_states).issubset(self.states):
            raise ValueError("All absorbing states must be in states")
        off_diag = q.copy()
        np.fill_diagonal(off_diag, 0.0)
        if np.any(off_diag < -1e-12):
            raise ValueError("CTMC off-diagonal generator rates must be nonnegative")
        if not np.allclose(q.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError("Every CTMC generator row must sum to zero")
        for name in self.absorbing_states:
            idx = self.states.index(name)
            if not np.allclose(q[idx], 0.0, atol=1e-12):
                raise ValueError(f"Absorbing state {name!r} must have a zero generator row")
        object.__setattr__(self, "generator", q)

    @property
    def transient_states(self) -> tuple[str, ...]:
        absorbing = set(self.absorbing_states)
        return tuple(state for state in self.states if state not in absorbing)

    def transition_matrix(self, elapsed: float) -> np.ndarray:
        if elapsed < 0:
            raise ValueError("elapsed must be nonnegative")
        p = expm(self.generator * float(elapsed))
        p[np.abs(p) < 1e-15] = 0.0
        return p

    def expected_transient_time(self, start_state: str) -> dict[str, float]:
        if start_state not in self.transient_states:
            raise ValueError("start_state must be transient")
        transient_indices = [self.states.index(s) for s in self.transient_states]
        t = self.generator[np.ix_(transient_indices, transient_indices)]
        fundamental = np.linalg.inv(-t)
        row = self.transient_states.index(start_state)
        return {
            state: float(fundamental[row, col])
            for col, state in enumerate(self.transient_states)
        }

    def absorption_probabilities(self, start_state: str) -> dict[str, float]:
        if start_state not in self.transient_states:
            raise ValueError("start_state must be transient")
        transient_indices = [self.states.index(s) for s in self.transient_states]
        absorbing_indices = [self.states.index(s) for s in self.absorbing_states]
        t = self.generator[np.ix_(transient_indices, transient_indices)]
        r = self.generator[np.ix_(transient_indices, absorbing_indices)]
        b = np.linalg.inv(-t) @ r
        row = self.transient_states.index(start_state)
        probs = np.clip(b[row], 0.0, 1.0)
        probs = probs / probs.sum()
        return {state: float(probs[i]) for i, state in enumerate(self.absorbing_states)}

    def expected_time_to_absorption(self, start_state: str) -> float:
        return float(sum(self.expected_transient_time(start_state).values()))


def make_reference_patient_flow_ctmc() -> CTMCModel:
    """Reference adult-flow CTMC used for deterministic tests and synthetic replay.

    States: ED -> Ward / ICU / discharge, Ward <-> ICU/stepdown, Stepdown -> discharge,
    and transfer as a second absorbing state.
    """
    states = ("ED", "Ward", "ICU", "Stepdown", "Discharged", "Transferred")
    q = np.zeros((6, 6), dtype=float)

    # ED mean event time ~4h; competing destinations.
    q[0, 1] = 0.145
    q[0, 2] = 0.035
    q[0, 4] = 0.065
    q[0, 5] = 0.005
    # Ward mean event time ~30h.
    q[1, 2] = 0.004
    q[1, 3] = 0.006
    q[1, 4] = 0.021
    q[1, 5] = 0.002
    # ICU mean event time ~42h.
    q[2, 1] = 0.006
    q[2, 3] = 0.010
    q[2, 4] = 0.005
    q[2, 5] = 0.003
    # Stepdown mean event time ~24h.
    q[3, 1] = 0.010
    q[3, 4] = 0.030
    q[3, 5] = 0.002

    for i in range(4):
        q[i, i] = -q[i].sum()
    return CTMCModel(states=states, generator=q, absorbing_states=("Discharged", "Transferred"))


def simulate_population_counts(
    model: CTMCModel,
    *,
    initial_counts: Sequence[int],
    arrivals_by_step: Sequence[int],
    arrival_state: str = "ED",
    step_hours: float = 1.0,
    seed: int = 20260817,
) -> np.ndarray:
    """Simulate aggregate CTMC counts using multinomial transitions.

    Returns an array of shape (steps + 1, n_states). New arrivals are inserted
    at the start of each step before transitions occur. Population conservation
    holds after accounting for those arrivals because absorbing states are kept
    explicitly in the state vector.
    """
    counts = np.asarray(initial_counts, dtype=int)
    if counts.shape != (len(model.states),) or np.any(counts < 0):
        raise ValueError("initial_counts must be one nonnegative integer per state")
    arrivals = np.asarray(arrivals_by_step, dtype=int)
    if arrivals.ndim != 1 or np.any(arrivals < 0):
        raise ValueError("arrivals_by_step must be nonnegative integers")
    p = model.transition_matrix(step_hours)
    if np.any(p < -1e-10) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("CTMC transition matrix is not stochastic")

    rng = np.random.default_rng(seed)
    arrival_index = model.states.index(arrival_state)
    trajectory = np.zeros((len(arrivals) + 1, len(model.states)), dtype=int)
    trajectory[0] = counts
    current = counts.copy()
    for t, new_arrivals in enumerate(arrivals, start=1):
        current[arrival_index] += int(new_arrivals)
        next_counts = np.zeros_like(current)
        for state_index, population in enumerate(current):
            if population:
                next_counts += rng.multinomial(int(population), p[state_index])
        current = next_counts
        trajectory[t] = current
    return trajectory
