from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config.dataclasses import STATE_ORDER, TransitionBundle


def get_state_order() -> List[str]:
    """Return the canonical Markov state order."""
    return STATE_ORDER.copy()


def get_state_index_map() -> Dict[str, int]:
    """Return mapping from state name to matrix index."""
    return {state: idx for idx, state in enumerate(STATE_ORDER)}


def validate_transition_dataframe(
    df: pd.DataFrame,
    cohort: str,
    tolerance: float = 1e-8,
) -> None:
    """
    Validate a long-format transition DataFrame.

    Checks:
    - required columns exist
    - all states are valid
    - probabilities lie in [0, 1]
    - each from_state has exactly one row for each to_state
    - each from_state row sums to 1
    """
    required_cols = ["from_state", "to_state", "prob"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Transition data for cohort '{cohort}' is missing columns: {missing_cols}"
        )

    valid_states = set(STATE_ORDER)

    bad_from_states = sorted(set(df["from_state"]) - valid_states)
    if bad_from_states:
        raise ValueError(
            f"Transition data for cohort '{cohort}' has invalid from_state values: "
            f"{bad_from_states}"
        )

    bad_to_states = sorted(set(df["to_state"]) - valid_states)
    if bad_to_states:
        raise ValueError(
            f"Transition data for cohort '{cohort}' has invalid to_state values: "
            f"{bad_to_states}"
        )

    if ((df["prob"] < 0) | (df["prob"] > 1)).any():
        raise ValueError(
            f"Transition data for cohort '{cohort}' contains probabilities outside [0, 1]"
        )

    pair_counts = (
        df.groupby(["from_state", "to_state"])
        .size()
        .reset_index(name="count")
    )
    duplicate_pairs = pair_counts[pair_counts["count"] != 1]
    if not duplicate_pairs.empty:
        raise ValueError(
            f"Transition data for cohort '{cohort}' must contain exactly one row per "
            f"(from_state, to_state) pair. Problematic pairs:\n{duplicate_pairs}"
        )

    expected_num_pairs = len(STATE_ORDER) * len(STATE_ORDER)
    if len(df) != expected_num_pairs:
        raise ValueError(
            f"Transition data for cohort '{cohort}' must have exactly "
            f"{expected_num_pairs} rows, but found {len(df)}"
        )

    row_sums = df.groupby("from_state")["prob"].sum().reindex(STATE_ORDER)
    missing_from_states = row_sums[row_sums.isna()].index.tolist()
    if missing_from_states:
        raise ValueError(
            f"Transition data for cohort '{cohort}' is missing transitions for "
            f"from_state values: {missing_from_states}"
        )

    bad_row_sums = row_sums[np.abs(row_sums - 1.0) > tolerance]
    if not bad_row_sums.empty:
        raise ValueError(
            f"Transition row sums for cohort '{cohort}' are not 1 within tolerance "
            f"{tolerance}:\n{bad_row_sums}"
        )


def transition_df_to_matrix(
    df: pd.DataFrame,
    cohort: str,
) -> np.ndarray:
    """
    Convert a validated long-format transition DataFrame to a NumPy matrix.

    Returns
    -------
    np.ndarray
        Square transition matrix P with state order defined by STATE_ORDER.
    """
    validate_transition_dataframe(df=df, cohort=cohort)

    state_to_idx = get_state_index_map()
    n_states = len(STATE_ORDER)
    matrix = np.zeros((n_states, n_states), dtype=float)

    for row in df.itertuples(index=False):
        i = state_to_idx[row.from_state]
        j = state_to_idx[row.to_state]
        matrix[i, j] = float(row.prob)

    return matrix


def transition_matrix_to_dataframe(
    matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Convert a transition matrix back to long-format DataFrame using STATE_ORDER.
    """
    n_states = len(STATE_ORDER)
    if matrix.shape != (n_states, n_states):
        raise ValueError(
            f"Expected matrix shape {(n_states, n_states)}, got {matrix.shape}"
        )

    rows = []
    for i, from_state in enumerate(STATE_ORDER):
        for j, to_state in enumerate(STATE_ORDER):
            rows.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "prob": float(matrix[i, j]),
                }
            )

    return pd.DataFrame(rows)


def load_transition_matrices(
    transition_bundle: TransitionBundle,
) -> Dict[str, np.ndarray]:
    """
    Load and validate all cohort transition matrices from a TransitionBundle.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping cohort -> transition matrix
    """
    matrices: Dict[str, np.ndarray] = {}

    for cohort, transition_data in transition_bundle.matrices.items():
        matrices[cohort] = transition_df_to_matrix(
            df=transition_data.df,
            cohort=cohort,
        )

    return matrices


def get_resource_state_indices() -> Dict[str, int]:
    """
    Return the state indices corresponding to occupancy-driving resources.

    ICU demand comes from state 'ICU'
    Ward demand comes from state 'Ward'
    """
    state_to_idx = get_state_index_map()
    return {
        "ICU": state_to_idx["ICU"],
        "Ward": state_to_idx["Ward"],
    }


def summarize_transition_matrix(
    matrix: np.ndarray,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return a readable DataFrame version of the matrix and its row sums.
    Useful for debugging and sanity checks.
    """
    df_matrix = pd.DataFrame(matrix, index=STATE_ORDER, columns=STATE_ORDER)
    row_sums = df_matrix.sum(axis=1)
    return df_matrix, row_sums