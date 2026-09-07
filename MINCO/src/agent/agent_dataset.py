from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "results/scenario_suite_summary.csv"
DEFAULT_OUTPUT = "results/agent_policy_dataset.csv"


POLICY_ORDER = [
    "optimized_network",
    "myopic_milp",
    "no_transfer",
    "local_only",
    "no_control",
]


def load_scenario_summary(csv_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    return df


def build_agent_dataset(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert scenario-suite summary into an agent-facing policy evaluation dataset.

    One row = one (scenario, policy) pair with the mean performance metrics
    that the agent can use to compare policies.
    """
    required_cols = [
        "scenario",
        "policy_name",
        "total_unsafe_excess_mean",
        "max_utilization_ratio_mean",
        "num_unsafe_rows_mean",
        "total_blocked_arrivals_mean",
    ]
    missing = [c for c in required_cols if c not in summary_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = summary_df[required_cols].copy()

    df = df.rename(
        columns={
            "policy_name": "policy",
            "total_unsafe_excess_mean": "unsafe_excess",
            "max_utilization_ratio_mean": "max_utilization",
            "num_unsafe_rows_mean": "unsafe_rows",
            "total_blocked_arrivals_mean": "blocked_arrivals",
        }
    )

    df["policy_rank_hint"] = df["policy"].apply(
        lambda x: POLICY_ORDER.index(x) if x in POLICY_ORDER else len(POLICY_ORDER)
    )

    df = df.sort_values(["scenario", "policy_rank_hint", "policy"]).reset_index(drop=True)
    return df


def save_agent_dataset(
    df: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    summary_df = load_scenario_summary()
    agent_df = build_agent_dataset(summary_df)
    output_path = save_agent_dataset(agent_df)

    print("\nAGENT POLICY DATASET")
    print(agent_df)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()