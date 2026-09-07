from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config.dataclasses import HealthcareInstance, STATE_ORDER
from src.markov.transition_utils import load_transition_matrices


def build_transition_matrix_lookup(
    instance: HealthcareInstance,
) -> Dict[str, np.ndarray]:
    """
    Return cohort -> transition matrix lookup.
    """
    return load_transition_matrices(instance.transitions)


def sample_next_state_counts(
    current_counts: np.ndarray,
    transition_matrix: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample realized next-state counts from current state counts and a transition matrix.

    Parameters
    ----------
    current_counts:
        Integer state-count vector in STATE_ORDER.
    transition_matrix:
        Row-stochastic transition matrix matching STATE_ORDER.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        Integer realized next-state counts in STATE_ORDER.
    """
    if rng is None:
        rng = np.random.default_rng()

    current_counts = np.asarray(current_counts, dtype=int)
    n_states = len(current_counts)

    next_counts = np.zeros(n_states, dtype=int)

    for i in range(n_states):
        count_in_state = int(current_counts[i])
        if count_in_state <= 0:
            continue

        probs = np.asarray(transition_matrix[i], dtype=float)
        sampled = rng.multinomial(count_in_state, probs)
        next_counts += sampled

    return next_counts


def inject_arrivals_into_ed(
    state_counts: np.ndarray,
    arrivals: int,
) -> np.ndarray:
    """
    Add realized arrivals into the ED state.
    """
    updated = np.asarray(state_counts, dtype=int).copy()
    ed_index = STATE_ORDER.index("ED")
    updated[ed_index] += int(arrivals)
    return updated


def sample_one_step_with_arrivals(
    current_counts: np.ndarray,
    arrivals: int,
    transition_matrix: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add arrivals into ED, then sample one realized Markov step.
    """
    with_arrivals = inject_arrivals_into_ed(
        state_counts=current_counts,
        arrivals=arrivals,
    )

    next_counts = sample_next_state_counts(
        current_counts=with_arrivals,
        transition_matrix=transition_matrix,
        rng=rng,
    )

    return next_counts


def format_state_count_vector(
    counts: np.ndarray,
) -> pd.DataFrame:
    """
    Convert a state-count vector into a tidy DataFrame.
    """
    counts = np.asarray(counts, dtype=int)

    return pd.DataFrame(
        {
            "state": STATE_ORDER,
            "count": counts,
        }
    )


def sample_cohort_paths_over_horizon(
    arrivals_by_day: pd.Series,
    transition_matrix: np.ndarray,
    initial_counts: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Sample realized state-count evolution for one cohort over a time horizon.

    Parameters
    ----------
    arrivals_by_day:
        Pandas Series indexed by day with integer arrivals.
    transition_matrix:
        Cohort-specific transition matrix.
    initial_counts:
        Optional starting state-count vector in STATE_ORDER.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    pd.DataFrame
        Columns:
            day, state, count
    """
    if rng is None:
        rng = np.random.default_rng()

    if initial_counts is None:
        current_counts = np.zeros(len(STATE_ORDER), dtype=int)
    else:
        current_counts = np.asarray(initial_counts, dtype=int).copy()

    records = []

    for day, arrivals in arrivals_by_day.items():
        current_counts = sample_one_step_with_arrivals(
            current_counts=current_counts,
            arrivals=int(arrivals),
            transition_matrix=transition_matrix,
            rng=rng,
        )

        for state, count in zip(STATE_ORDER, current_counts):
            records.append(
                {
                    "day": int(day),
                    "state": state,
                    "count": int(count),
                }
            )

    return pd.DataFrame(records)