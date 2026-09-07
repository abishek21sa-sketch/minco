from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASET = "results/agent_policy_dataset.csv"


@dataclass
class AgentState:
    scenario: str
    forecast_max_utilization: float
    forecast_unsafe_excess: float
    forecast_unsafe_rows: float
    transfer_available: bool = True


def load_agent_dataset(
    csv_path: str | Path = DEFAULT_DATASET,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    return df


def choose_best_policy_from_dataset(
    dataset_df: pd.DataFrame,
    scenario: str,
    weight_unsafe_excess: float = 1.0,
    weight_unsafe_rows: float = 2.0,
    weight_blocked_arrivals: float = 0.75,
    weight_utilization: float = 1.0,
) -> pd.DataFrame:
    """
    Score each policy within a scenario and return ranked policies.

    Lower score is better.
    """
    sub = dataset_df[dataset_df["scenario"] == scenario].copy()
    if sub.empty:
        raise ValueError(f"No rows found for scenario '{scenario}'")

    sub["score"] = (
        weight_unsafe_excess * sub["unsafe_excess"]
        + weight_unsafe_rows * sub["unsafe_rows"]
        + weight_blocked_arrivals * sub["blocked_arrivals"]
        + weight_utilization * sub["max_utilization"]
    )

    sub = sub.sort_values(["score", "unsafe_excess", "blocked_arrivals"]).reset_index(drop=True)
    return sub


def recommend_policy_rule_based(
    state: AgentState,
) -> dict[str, Any]:
    """
    First rule-based agent.

    Logic:
    - very high overload risk -> optimized_network
    - moderate overload risk -> myopic_milp
    - transfer unavailable -> no_transfer
    - mild condition -> local_only

    This is intentionally simple and interpretable.
    """
    if not state.transfer_available or state.scenario == "transfer_disabled":
        recommended = "no_transfer"
        reason = "Transfers are unavailable, so the policy must rely on local capacity actions."
    elif state.forecast_max_utilization >= 1.50 or state.forecast_unsafe_excess >= 12:
        recommended = "optimized_network"
        reason = (
            "Severe predicted overload calls for the strongest network-wide response, "
            "including proactive coordination."
        )
    elif state.forecast_max_utilization >= 1.20 or state.forecast_unsafe_rows >= 2:
        recommended = "myopic_milp"
        reason = (
            "Moderate predicted overload suggests using optimization, but not necessarily "
            "the full network-wide preventive policy."
        )
    else:
        recommended = "local_only"
        reason = "Mild predicted conditions suggest that local control is sufficient."

    return {
        "recommended_policy": recommended,
        "reason": reason,
        "state": state.__dict__,
    }


def recommend_policy_with_dataset_support(
    state: AgentState,
    dataset_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Hybrid agent:
    1. use transparent rule logic
    2. also show dataset-based ranking for the scenario
    """
    rule_result = recommend_policy_rule_based(state)

    try:
        ranked = choose_best_policy_from_dataset(
            dataset_df=dataset_df,
            scenario=state.scenario,
        )
        top_ranked = ranked.iloc[0]["policy"]
        top_score = float(ranked.iloc[0]["score"])
        ranking_preview = ranked[
            ["policy", "unsafe_excess", "unsafe_rows", "blocked_arrivals", "max_utilization", "score"]
        ].copy()
    except Exception as exc:
        top_ranked = None
        top_score = None
        ranking_preview = pd.DataFrame()
        rule_result["dataset_error"] = str(exc)

    rule_result["dataset_top_policy"] = top_ranked
    rule_result["dataset_top_score"] = top_score
    rule_result["dataset_ranking"] = ranking_preview

    return rule_result


def main() -> None:
    dataset_df = load_agent_dataset()

    demo_states = [
        AgentState(
            scenario="baseline",
            forecast_max_utilization=1.05,
            forecast_unsafe_excess=4.5,
            forecast_unsafe_rows=2.0,
            transfer_available=True,
        ),
        AgentState(
            scenario="h3_c4_shock_2p0",
            forecast_max_utilization=1.28,
            forecast_unsafe_excess=9.0,
            forecast_unsafe_rows=3.0,
            transfer_available=True,
        ),
        AgentState(
            scenario="h3_icu_capacity_reduced",
            forecast_max_utilization=1.78,
            forecast_unsafe_excess=15.0,
            forecast_unsafe_rows=4.0,
            transfer_available=True,
        ),
        AgentState(
            scenario="transfer_disabled",
            forecast_max_utilization=1.10,
            forecast_unsafe_excess=5.5,
            forecast_unsafe_rows=2.0,
            transfer_available=False,
        ),
    ]

    for state in demo_states:
        result = recommend_policy_with_dataset_support(state, dataset_df)

        print("\n==============================")
        print(f"SCENARIO: {state.scenario}")
        print("==============================")
        print("Rule recommendation:", result["recommended_policy"])
        print("Reason:", result["reason"])
        print("Dataset top policy:", result["dataset_top_policy"])

        if not result["dataset_ranking"].empty:
            print("\nTop ranking preview:")
            print(result["dataset_ranking"].head(5))


if __name__ == "__main__":
    main()