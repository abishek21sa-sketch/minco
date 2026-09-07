from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.rl.q_learning_agent import (
    train_q_learning_agent,
    benchmark_rl_vs_fixed_policies,
)


RESULTS_DIR = Path("results")
RL_DIR = RESULTS_DIR / "rl"
RL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper summaries
# ============================================================

def build_controller_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average reward by controller / policy.
    Higher reward is better.
    """

    out = (
        df.groupby(
            ["controller_type", "policy_name"],
            as_index=False,
        )
        .agg(
            mean_reward=("episode_reward", "mean"),
            std_reward=("episode_reward", "std"),
            n=("episode_reward", "count"),
        )
        .sort_values("mean_reward", ascending=False)
        .reset_index(drop=True)
    )

    return out


def build_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reward summary by starting regime.
    """

    out = (
        df.groupby(
            ["start_regime", "controller_type", "policy_name"],
            as_index=False,
        )
        .agg(
            mean_reward=("episode_reward", "mean"),
            std_reward=("episode_reward", "std"),
            n=("episode_reward", "count"),
        )
        .sort_values(
            ["start_regime", "mean_reward"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return out


def build_headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best controller in each regime.
    """

    rows = []

    for regime in sorted(df["start_regime"].unique()):
        sub = df[df["start_regime"] == regime].copy()

        best = sub.sort_values(
            "episode_reward",
            ascending=False,
        ).iloc[0]

        rows.append(
            {
                "start_regime": regime,
                "best_policy": best["policy_name"],
                "controller_type": best["controller_type"],
                "best_reward": best["episode_reward"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    print("\nRL VS OPTIMIZER BENCHMARK")
    print("=" * 70, flush=True)

    # --------------------------------------------------------
    # Train RL agent
    # --------------------------------------------------------

    print("\nTraining RL agent...\n", flush=True)

    agent, history_df = train_q_learning_agent(
        n_episodes=25,
        use_small_encoder=True,
        env_kwargs={
            "n_replications_per_step": 2,
            "max_steps": 3,
            "random_seed": 42,
        },
        agent_kwargs={
            "alpha": 0.10,
            "gamma": 0.95,
            "epsilon": 1.00,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.96,
            "random_seed": 42,
        },
    )

    print("\nTraining complete.", flush=True)

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("\nRunning controller benchmark...\n", flush=True)

    summary_df, rollout_store = benchmark_rl_vs_fixed_policies(
        agent=agent,
        use_small_encoder=True,
        env_kwargs={
            "n_replications_per_step": 2,
            "max_steps": 3,
            "random_seed": 99,
        },
        evaluation_regimes=[
            "normal",
            "surge",
            "crisis",
        ],
    )

    print("\nBenchmark complete.", flush=True)

    # --------------------------------------------------------
    # Derived tables
    # --------------------------------------------------------

    controller_summary = build_controller_summary(summary_df)
    regime_summary = build_regime_summary(summary_df)
    headline = build_headline_table(summary_df)
    q_table_df = agent.to_dataframe()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    history_path = RL_DIR / "rl_optimizer_training_history.csv"
    benchmark_path = RL_DIR / "rl_optimizer_benchmark_full.csv"
    controller_path = RL_DIR / "rl_optimizer_controller_summary.csv"
    regime_path = RL_DIR / "rl_optimizer_regime_summary.csv"
    headline_path = RL_DIR / "rl_optimizer_headline.csv"
    qtable_path = RL_DIR / "rl_optimizer_q_table.csv"
    model_path = RL_DIR / "rl_optimizer_agent.pkl"
    meta_path = RL_DIR / "rl_optimizer_run_summary.json"

    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(benchmark_path, index=False)
    controller_summary.to_csv(controller_path, index=False)
    regime_summary.to_csv(regime_path, index=False)
    headline.to_csv(headline_path, index=False)
    q_table_df.to_csv(qtable_path, index=False)

    agent.save_pickle(model_path)

    for rollout_name, rollout_df in rollout_store.items():
        rollout_path = RL_DIR / f"{rollout_name}__optimizer_rollout.csv"
        rollout_df.to_csv(rollout_path, index=False)

    meta = {
        "n_training_episodes": int(len(history_df)),
        "n_states_in_q_table": int(len(q_table_df)),
        "n_benchmark_rows": int(len(summary_df)),
        "best_overall_policy": (
            controller_summary.iloc[0]["policy_name"]
            if not controller_summary.empty
            else None
        ),
        "best_overall_mean_reward": (
            float(controller_summary.iloc[0]["mean_reward"])
            if not controller_summary.empty
            else None
        ),
    }

    meta_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print outputs
    # --------------------------------------------------------

    print("\n=== HEADLINE WINNERS ===")
    print(headline)

    print("\n=== CONTROLLER SUMMARY ===")
    print(controller_summary)

    print("\n=== REGIME SUMMARY ===")
    print(regime_summary)

    print("\nSaved:")
    print(history_path)
    print(benchmark_path)
    print(controller_path)
    print(regime_path)
    print(headline_path)
    print(qtable_path)
    print(model_path)
    print(meta_path)


if __name__ == "__main__":
    main()