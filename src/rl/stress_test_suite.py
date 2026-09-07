from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd

from src.rl.hospital_env import HospitalRLEnv
from src.rl.q_learning_agent_v2 import QLearningAgentV2
from src.rl.state_encoder import encode_state_to_key_small


RESULTS_DIR = Path("results")
RL_DIR = RESULTS_DIR / "rl"
RL_STRESS_DIR = RL_DIR / "stress_test"
RL_STRESS_DIR.mkdir(parents=True, exist_ok=True)


OUTPUT_EPISODE_PATH = RL_STRESS_DIR / "rl_stress_episodes.csv"
OUTPUT_SUMMARY_PATH = RL_STRESS_DIR / "rl_stress_summary.csv"
OUTPUT_BENCHMARK_PATH = RL_STRESS_DIR / "rl_stress_fixed_policy_benchmark.csv"
OUTPUT_HEADLINE_PATH = RL_STRESS_DIR / "rl_stress_headline.csv"
OUTPUT_JSON_PATH = RL_STRESS_DIR / "rl_stress_report.json"


STRESS_CONFIGS = [
    {
        "name": "baseline",
        "n_episodes": 30,
        "max_steps": 3,
        "n_replications_per_step": 1,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.98,
        "start_regime_cycle": ["normal", "surge", "crisis"],
    },
    {
        "name": "long_horizon",
        "n_episodes": 40,
        "max_steps": 5,
        "n_replications_per_step": 1,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.985,
        "start_regime_cycle": ["normal", "surge", "crisis"],
    },
    {
        "name": "low_exploration",
        "n_episodes": 30,
        "max_steps": 3,
        "n_replications_per_step": 1,
        "epsilon_start": 0.4,
        "epsilon_decay": 0.99,
        "start_regime_cycle": ["normal", "surge", "crisis"],
    },
    {
        "name": "high_exploration",
        "n_episodes": 30,
        "max_steps": 3,
        "n_replications_per_step": 1,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.995,
        "start_regime_cycle": ["normal", "surge", "crisis"],
    },
    {
        "name": "crisis_start_bias",
        "n_episodes": 40,
        "max_steps": 4,
        "n_replications_per_step": 1,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.985,
        "start_regime_cycle": ["crisis"],
    },
]


FIXED_POLICIES = [
    "local_only",
    "myopic_milp",
    "optimized_network",
    "robust_optimized_network",
    "regime_robust_optimized_network",
]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def shape_reward(
    raw_reward: float,
    next_state: dict[str, Any],
    *,
    crisis_penalty: float = 2.0,
    high_util_penalty: float = 1.0,
    blocked_penalty_scale: float = 0.02,
) -> float:
    reward = float(raw_reward)

    regime = str(next_state.get("regime", "normal")).lower()
    util = safe_float(next_state.get("prev_max_utilization"), 1.0)
    blocked = safe_float(next_state.get("prev_blocked_arrivals"), 0.0)

    if regime == "crisis":
        reward -= crisis_penalty

    if util >= 1.10:
        reward -= high_util_penalty

    reward -= blocked_penalty_scale * blocked

    return reward


def run_single_config(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, QLearningAgentV2]:
    name = config["name"]

    print(f"\n=== Running config: {name} ===", flush=True)

    env = HospitalRLEnv(
        n_replications_per_step=config.get("n_replications_per_step", 1),
        max_steps=config.get("max_steps", 3),
        random_seed=42,
    )

    agent = QLearningAgentV2(
        action_names=env.get_action_space(),
        epsilon=config.get("epsilon_start", 1.0),
        epsilon_decay=config.get("epsilon_decay", 0.98),
        epsilon_min=0.05,
        alpha=0.10,
        gamma=0.95,
        optimistic_q_init=1.0,
        random_seed=42,
    )

    start_regime_cycle = config.get("start_regime_cycle", ["normal", "surge", "crisis"])

    episode_rows = []

    for ep in range(config["n_episodes"]):
        start_regime = start_regime_cycle[ep % len(start_regime_cycle)]

        state = env.reset(start_regime=start_regime)
        state_key = encode_state_to_key_small(state)

        total_raw_reward = 0.0
        total_shaped_reward = 0.0
        actions_taken = []

        done = False
        step_count = 0

        while not done:
            action_idx = agent.select_action(state_key, explore=True)
            action_name = env.idx_to_action[action_idx]

            result = env.step(action_idx)

            next_state = result.next_state
            next_state_key = encode_state_to_key_small(next_state)

            raw_reward = float(result.reward)
            shaped_reward = shape_reward(raw_reward, next_state)

            agent.update(
                state_key=state_key,
                action_idx=action_idx,
                reward=shaped_reward,
                next_state_key=next_state_key,
                done=result.done,
            )

            total_raw_reward += raw_reward
            total_shaped_reward += shaped_reward
            actions_taken.append(action_name)

            state_key = next_state_key
            done = result.done
            step_count += 1

        agent.decay_epsilon()

        episode_rows.append(
            {
                "config": name,
                "episode": ep + 1,
                "start_regime": start_regime,
                "total_raw_reward": total_raw_reward,
                "total_shaped_reward": total_shaped_reward,
                "final_epsilon": agent.epsilon,
                "episode_steps": step_count,
                "actions_taken": ",".join(actions_taken),
            }
        )

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"{name}: episode {ep + 1}/{config['n_episodes']} | "
                f"raw={total_raw_reward:.3f} | shaped={total_shaped_reward:.3f} | "
                f"eps={agent.epsilon:.3f}",
                flush=True,
            )

    ep_df = pd.DataFrame(episode_rows)

    summary = (
        ep_df.groupby("config", as_index=False)
        .agg(
            mean_raw_reward=("total_raw_reward", "mean"),
            std_raw_reward=("total_raw_reward", "std"),
            mean_shaped_reward=("total_shaped_reward", "mean"),
            std_shaped_reward=("total_shaped_reward", "std"),
            min_raw_reward=("total_raw_reward", "min"),
            max_raw_reward=("total_raw_reward", "max"),
            final_epsilon=("final_epsilon", "last"),
            n=("episode", "count"),
        )
    )

    return ep_df, summary, agent


def evaluate_agent(
    agent: QLearningAgentV2,
    *,
    config_name: str,
    start_regimes: list[str],
    seeds: list[int],
    max_steps: int,
    n_replications_per_step: int,
) -> pd.DataFrame:
    rows = []

    for regime in start_regimes:
        for seed in seeds:
            env = HospitalRLEnv(
                n_replications_per_step=n_replications_per_step,
                max_steps=max_steps,
                random_seed=seed,
            )

            state = env.reset(start_regime=regime)
            state_key = encode_state_to_key_small(state)

            done = False
            total_reward = 0.0
            actions_taken = []
            step_count = 0

            while not done:
                action_idx = agent.greedy_action(state_key)
                action_name = env.idx_to_action[action_idx]

                result = env.step(action_idx)

                total_reward += float(result.reward)
                actions_taken.append(action_name)

                state_key = encode_state_to_key_small(result.next_state)
                done = result.done
                step_count += 1

            rows.append(
                {
                    "config": config_name,
                    "controller_type": "rl_policy",
                    "policy_name": "learned_q_policy",
                    "start_regime": regime,
                    "seed": seed,
                    "episode_reward": total_reward,
                    "episode_steps": step_count,
                    "actions_taken": ",".join(actions_taken),
                }
            )

    return pd.DataFrame(rows)


def run_fixed_policy_benchmark(
    *,
    start_regimes: list[str],
    seeds: list[int],
    max_steps: int,
    n_replications_per_step: int,
) -> pd.DataFrame:
    print("\n=== Running fixed policy benchmark ===", flush=True)

    rows = []

    for policy in FIXED_POLICIES:
        for regime in start_regimes:
            for seed in seeds:
                env = HospitalRLEnv(
                    n_replications_per_step=n_replications_per_step,
                    max_steps=max_steps,
                    random_seed=seed,
                )

                rollout_df = env.run_fixed_policy_episode(policy)

                total_reward = (
                    float(rollout_df["reward"].sum())
                    if "reward" in rollout_df.columns
                    else 0.0
                )

                rows.append(
                    {
                        "controller_type": "fixed_policy",
                        "policy_name": policy,
                        "start_regime": regime,
                        "seed": seed,
                        "episode_reward": total_reward,
                        "episode_steps": len(rollout_df),
                        "actions_taken": policy,
                    }
                )

    return pd.DataFrame(rows)


def summarize_controller_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["controller_type", "policy_name"], as_index=False)
        .agg(
            mean_reward=("episode_reward", "mean"),
            std_reward=("episode_reward", "std"),
            min_reward=("episode_reward", "min"),
            max_reward=("episode_reward", "max"),
            n=("episode_reward", "count"),
        )
        .sort_values("mean_reward", ascending=False)
        .reset_index(drop=True)
    )


def build_headline(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    ranked = summary_df.sort_values("mean_reward", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    best = ranked.iloc[0]
    rl_rows = ranked[ranked["policy_name"] == "learned_q_policy"]

    if rl_rows.empty:
        rl_rank = None
        rl_mean = None
        gap = None
    else:
        rl_row = rl_rows.iloc[0]
        rl_rank = int(rl_row["rank"])
        rl_mean = float(rl_row["mean_reward"])
        gap = rl_mean - float(best["mean_reward"])

    return pd.DataFrame(
        [
            {
                "best_policy": best["policy_name"],
                "best_controller_type": best["controller_type"],
                "best_mean_reward": float(best["mean_reward"]),
                "rl_rank": rl_rank,
                "rl_mean_reward": rl_mean,
                "gap_rl_minus_best": gap,
                "n_controllers": len(ranked),
            }
        ]
    )


def main() -> None:
    print("\nRL STRESS TEST SUITE")
    print("=" * 70, flush=True)

    all_episodes = []
    all_eval_rows = []
    all_training_summaries = []

    start_regimes = ["normal", "surge", "crisis"]
    eval_seeds = [101, 202, 303]

    for config in STRESS_CONFIGS:
        ep_df, train_summary, agent = run_single_config(config)

        all_episodes.append(ep_df)
        all_training_summaries.append(train_summary)

        eval_df = evaluate_agent(
            agent,
            config_name=config["name"],
            start_regimes=start_regimes,
            seeds=eval_seeds,
            max_steps=config.get("max_steps", 3),
            n_replications_per_step=config.get("n_replications_per_step", 1),
        )

        all_eval_rows.append(eval_df)

    episode_df = pd.concat(all_episodes, ignore_index=True)
    training_summary_df = pd.concat(all_training_summaries, ignore_index=True)
    rl_eval_df = pd.concat(all_eval_rows, ignore_index=True)

    fixed_df = run_fixed_policy_benchmark(
        start_regimes=start_regimes,
        seeds=eval_seeds,
        max_steps=3,
        n_replications_per_step=1,
    )

    combined_eval_df = pd.concat([rl_eval_df, fixed_df], ignore_index=True)
    controller_summary = summarize_controller_results(combined_eval_df)
    headline_df = build_headline(controller_summary)

    episode_df.to_csv(OUTPUT_EPISODE_PATH, index=False)
    training_summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    fixed_df.to_csv(OUTPUT_BENCHMARK_PATH, index=False)
    headline_df.to_csv(OUTPUT_HEADLINE_PATH, index=False)

    combined_eval_path = RL_STRESS_DIR / "rl_stress_combined_eval.csv"
    controller_summary_path = RL_STRESS_DIR / "rl_stress_controller_summary.csv"

    combined_eval_df.to_csv(combined_eval_path, index=False)
    controller_summary.to_csv(controller_summary_path, index=False)

    report = {
        "n_configs": len(STRESS_CONFIGS),
        "configs": [c["name"] for c in STRESS_CONFIGS],
        "n_eval_rows": int(len(combined_eval_df)),
        "best_policy": (
            str(headline_df.iloc[0]["best_policy"]) if not headline_df.empty else None
        ),
        "rl_rank": (
            int(headline_df.iloc[0]["rl_rank"]) if not headline_df.empty and pd.notna(headline_df.iloc[0]["rl_rank"]) else None
        ),
    }

    OUTPUT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Training Stress Summary ===")
    print(training_summary_df.to_string(index=False))

    print("\n=== Controller Summary ===")
    print(controller_summary.to_string(index=False))

    print("\n=== Headline ===")
    print(headline_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_EPISODE_PATH)
    print(OUTPUT_SUMMARY_PATH)
    print(OUTPUT_BENCHMARK_PATH)
    print(combined_eval_path)
    print(controller_summary_path)
    print(OUTPUT_HEADLINE_PATH)
    print(OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()