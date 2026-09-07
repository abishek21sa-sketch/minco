from __future__ import annotations

from typing import Any

import pandas as pd

from src.agent.rule_policy import (
    AgentState,
    load_agent_dataset,
    recommend_policy_rule_based,
    choose_best_policy_from_dataset,
)


def hybrid_policy_decision(
    state: AgentState,
    dataset_df: pd.DataFrame,
) -> dict[str, Any]:

    rule = recommend_policy_rule_based(state)
    ranked = choose_best_policy_from_dataset(dataset_df, state.scenario)

    dataset_top = ranked.iloc[0]["policy"]

    # --- Reconciliation logic ---
    if rule["recommended_policy"] == dataset_top:
        final_policy = dataset_top
        decision_type = "agreement"

    else:
        # TRUST DATA more in high-stress
        if state.forecast_max_utilization >= 1.4:
            final_policy = dataset_top
            decision_type = "data_override"

        # TRUST RULE more in low-stress
        elif state.forecast_max_utilization < 1.1:
            final_policy = rule["recommended_policy"]
            decision_type = "rule_override"

        # otherwise blend
        else:
            final_policy = dataset_top
            decision_type = "soft_override"

    return {
        "final_policy": final_policy,
        "decision_type": decision_type,
        "rule_policy": rule["recommended_policy"],
        "dataset_policy": dataset_top,
        "scenario": state.scenario,
    }


def main():
    df = load_agent_dataset()

    states = [
        AgentState("baseline", 1.05, 4.5, 2.0, True),
        AgentState("h3_c4_shock_2p0", 1.28, 9.0, 3.0, True),
        AgentState("h3_icu_capacity_reduced", 1.78, 15.0, 4.0, True),
        AgentState("transfer_disabled", 1.10, 5.5, 2.0, False),
    ]

    for s in states:
        result = hybrid_policy_decision(s, df)

        print("\n====================")
        print(f"Scenario: {s.scenario}")
        print("====================")
        print(result)


if __name__ == "__main__":
    main()