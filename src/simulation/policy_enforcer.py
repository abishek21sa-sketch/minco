from __future__ import annotations

from typing import Dict

import pandas as pd


def _build_elective_acceptance_lookup(
    policy_snapshot: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a hospital-day lookup table for elective c3 acceptance.

    Expected keys in policy_snapshot:
        elective_accepted
        elective_rejected

    Returns
    -------
    pd.DataFrame
        Columns:
            day, hospital_id, elective_accepted, elective_rejected
    """
    accepted = policy_snapshot["elective_accepted"].copy()
    rejected = policy_snapshot["elective_rejected"].copy()

    merged = accepted.merge(
        rejected,
        on=["day", "hospital_id"],
        how="outer",
        validate="one_to_one",
    )

    merged["elective_accepted"] = merged["elective_accepted"].fillna(0.0)
    merged["elective_rejected"] = merged["elective_rejected"].fillna(0.0)

    merged = merged.sort_values(["day", "hospital_id"]).reset_index(drop=True)
    return merged


def enforce_elective_policy_on_arrivals(
    realized_arrivals_df: pd.DataFrame,
    policy_snapshot: Dict[str, pd.DataFrame],
    elective_cohort: str = "c3",
) -> pd.DataFrame:
    """
    Enforce the elective policy on realized arrivals.

    Logic
    -----
    - Non-elective cohorts remain unchanged.
    - For the elective cohort (default c3), realized arrivals are capped by
      the accepted elective volume in the policy snapshot:
            enforced_realized_arrivals = min(realized_arrivals, elective_accepted)

    Returns
    -------
    pd.DataFrame
        Original realized arrivals table plus:
            enforced_realized_arrivals
            arrivals_blocked_by_policy
    """
    arrivals = realized_arrivals_df.copy()

    required_cols = {"day", "hospital_id", "cohort", "realized_arrivals"}
    missing = required_cols - set(arrivals.columns)
    if missing:
        raise ValueError(f"Missing required arrival columns: {sorted(missing)}")

    elective_lookup = _build_elective_acceptance_lookup(policy_snapshot)

    arrivals = arrivals.merge(
        elective_lookup,
        on=["day", "hospital_id"],
        how="left",
        validate="many_to_one",
    )

    arrivals["elective_accepted"] = arrivals["elective_accepted"].fillna(0.0).astype(float)
    arrivals["elective_rejected"] = arrivals["elective_rejected"].fillna(0.0).astype(float)
    arrivals["realized_arrivals"] = arrivals["realized_arrivals"].astype(float)

    arrivals["enforced_realized_arrivals"] = arrivals["realized_arrivals"].astype(float)

    elective_mask = arrivals["cohort"] == elective_cohort

    arrivals.loc[elective_mask, "enforced_realized_arrivals"] = arrivals.loc[
        elective_mask,
        ["realized_arrivals", "elective_accepted"],
    ].min(axis=1)

    arrivals["arrivals_blocked_by_policy"] = (
        arrivals["realized_arrivals"] - arrivals["enforced_realized_arrivals"]
    ).clip(lower=0.0)

    arrivals["enforced_realized_arrivals"] = arrivals["enforced_realized_arrivals"].round().astype(int)
    arrivals["arrivals_blocked_by_policy"] = arrivals["arrivals_blocked_by_policy"].round().astype(int)
    arrivals["realized_arrivals"] = arrivals["realized_arrivals"].round().astype(int)

    arrivals = arrivals.sort_values(
        ["day", "hospital_id", "cohort"]
    ).reset_index(drop=True)

    return arrivals

def summarize_enforced_arrivals(
    enforced_arrivals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize enforced arrivals and blocked arrivals by day and hospital.
    """
    summary = (
        enforced_arrivals_df.groupby(["day", "hospital_id"], as_index=False)[
            ["realized_arrivals", "enforced_realized_arrivals", "arrivals_blocked_by_policy"]
        ]
        .sum()
        .sort_values(["day", "hospital_id"])
        .reset_index(drop=True)
    )

    return summary


def extract_transfer_policy_table(
    policy_snapshot: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Return the ICU transfer policy table from the snapshot.

    This is not yet dynamically enforced inside the twin, but exposing it here
    keeps the interface ready for the next step.
    """
    expected_cols = [
        "from_hospital",
        "to_hospital",
        "day",
        "icu_transfer_load",
    ]

    transfer_df = policy_snapshot.get("icu_transfers")

    if transfer_df is None or transfer_df.empty:
        return pd.DataFrame(columns=expected_cols)

    transfer_df = transfer_df.copy()

    for col in expected_cols:
        if col not in transfer_df.columns:
            transfer_df[col] = pd.Series(dtype=float if col == "icu_transfer_load" else object)

    transfer_df = transfer_df[expected_cols].copy()

    return transfer_df.sort_values(
        ["day", "from_hospital", "to_hospital"]
    ).reset_index(drop=True)