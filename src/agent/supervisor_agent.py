from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.agent.hybrid_agent import hybrid_policy_decision
from src.agent.rule_policy import AgentState, load_agent_dataset


def classify_stress_level(state: AgentState) -> str:
    """
    Convert numeric forecast indicators into an interpretable stress label.
    """
    if state.forecast_max_utilization >= 1.50 or state.forecast_unsafe_excess >= 12:
        return "severe"
    if state.forecast_max_utilization >= 1.20 or state.forecast_unsafe_rows >= 2:
        return "moderate"
    return "mild"


def summarize_policy_tradeoff(
    ranked_df: pd.DataFrame,
    top_k: int = 3,
) -> str:
    """
    Build a concise natural-language summary of the top policy options.
    """
    if ranked_df.empty:
        return "No ranked policy comparison was available for this scenario."

    top = ranked_df.head(top_k).copy()
    parts = []

    for _, row in top.iterrows():
        parts.append(
            (
                f"{row['policy']} "
                f"(unsafe_excess={row['unsafe_excess']:.2f}, "
                f"unsafe_rows={row['unsafe_rows']:.2f}, "
                f"blocked_arrivals={row['blocked_arrivals']:.2f}, "
                f"max_utilization={row['max_utilization']:.3f})"
            )
        )

    return "Top policy ranking for this scenario: " + "; ".join(parts) + "."


def build_supervisor_message(
    state: AgentState,
    hybrid_result: dict[str, Any],
    ranked_df: pd.DataFrame,
) -> str:
    """
    Create a supervisor-facing explanation message.
    """
    stress = classify_stress_level(state)

    final_policy = hybrid_result["final_policy"]
    decision_type = hybrid_result["decision_type"]
    rule_policy = hybrid_result["rule_policy"]
    dataset_policy = hybrid_result["dataset_policy"]

    if decision_type == "agreement":
        decision_explainer = (
            f"Both the rule layer and the empirical policy ranking agree on {final_policy}."
        )
    elif decision_type == "rule_override":
        decision_explainer = (
            f"The final decision stays with the rule-based recommendation {rule_policy}, "
            f"even though the scenario ranking preferred {dataset_policy}. "
            f"This happened because the current operating condition is mild enough to avoid "
            f"an unnecessary escalation in control intensity."
        )
    elif decision_type == "data_override":
        decision_explainer = (
            f"The final decision overrides the rule-based recommendation {rule_policy} and switches "
            f"to {dataset_policy}, because the predicted stress is high and the empirical scenario "
            f"results suggest that {dataset_policy} performs better under this condition."
        )
    else:
        decision_explainer = (
            f"The final decision uses a soft override: the rule-based recommendation was {rule_policy}, "
            f"but the scenario data ranked {dataset_policy} higher, so the final policy is {final_policy}."
        )

    action_guidance = {
        "optimized_network": (
            "Activate the full network-wide response. Use proactive coordination, allow transfer-oriented "
            "rebalancing, and accept stronger access restrictions if needed to protect safety."
        ),
        "myopic_milp": (
            "Use the day-by-day optimization policy. This is a moderate-control response that preserves "
            "more access while still applying optimization to immediate overload."
        ),
        "no_transfer": (
            "Operate with local capacity actions only and do not rely on inter-hospital transfers. "
            "This is appropriate when transfers are unavailable or likely to be ineffective."
        ),
        "local_only": (
            "Use local hospital management without escalation to heavier network coordination. "
            "This is most appropriate under mild or contained stress."
        ),
        "no_control": (
            "This policy is not recommended operationally except as a benchmark. Avoid using it in practice."
        ),
    }.get(
        final_policy,
        "Apply the selected policy and monitor utilization and unsafe capacity indicators closely.",
    )

    tradeoff_summary = summarize_policy_tradeoff(ranked_df)

    message = (
        f"Supervisor decision summary:\n"
        f"- Scenario: {state.scenario}\n"
        f"- Stress level: {stress}\n"
        f"- Forecast max utilization: {state.forecast_max_utilization:.3f}\n"
        f"- Forecast unsafe excess: {state.forecast_unsafe_excess:.2f}\n"
        f"- Forecast unsafe rows: {state.forecast_unsafe_rows:.2f}\n"
        f"- Transfer availability: {state.transfer_available}\n"
        f"- Final recommended policy: {final_policy}\n"
        f"- Decision mode: {decision_type}\n\n"
        f"{decision_explainer}\n\n"
        f"Operational guidance: {action_guidance}\n\n"
        f"{tradeoff_summary}"
    )
    return message


def explain_supervisor_decision(
    state: AgentState,
    dataset_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Full supervisor-facing explanation wrapper.

    Returns:
    - raw state
    - hybrid result
    - ranked comparison table
    - final natural-language explanation
    """
    hybrid_result = hybrid_policy_decision(state, dataset_df)

    ranked_df = dataset_df[dataset_df["scenario"] == state.scenario].copy()
    ranked_df["score"] = (
        ranked_df["unsafe_excess"]
        + 2.0 * ranked_df["unsafe_rows"]
        + 0.75 * ranked_df["blocked_arrivals"]
        + ranked_df["max_utilization"]
    )
    ranked_df = ranked_df.sort_values(
        ["score", "unsafe_excess", "blocked_arrivals"]
    ).reset_index(drop=True)

    explanation = build_supervisor_message(
        state=state,
        hybrid_result=hybrid_result,
        ranked_df=ranked_df,
    )

    return {
        "state": asdict(state),
        "hybrid_result": hybrid_result,
        "ranking": ranked_df,
        "explanation": explanation,
    }


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
        result = explain_supervisor_decision(state, dataset_df)

        print("\n" + "=" * 70)
        print(result["explanation"])
        print("\nTop ranked policies:")
        print(
            result["ranking"][
                ["policy", "unsafe_excess", "unsafe_rows", "blocked_arrivals", "max_utilization", "score"]
            ].head(5)
        )


if __name__ == "__main__":
    main()