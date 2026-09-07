from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import pickle
import random
from typing import Callable, Any

import numpy as np
import pandas as pd

from src.rl.hospital_env import HospitalRLEnv
from src.rl.state_encoder import (
    encode_state_to_key,
    encode_state_to_key_small,
)


RESULTS_DIR = Path("results")
RL_DIR = RESULTS_DIR / "rl"
RL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================

DEFAULT_FIXED_POLICIES = [
    "local_only",
    "myopic_milp",
    "optimized_network",
    "robust_optimized_network",
    "regime_robust_optimized_network",
]

DEFAULT_EVAL_REGIMES = ["normal", "surge", "crisis"]


# ============================================================
# Helpers
# ============================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start:i + 1])))
    return out


# ============================================================
# Q-learning agent v2
# ============================================================

class QLearningAgentV2:
    def __init__(
        self,
        action_names: list[str],
        *,
        alpha: float = 0.08,
        gamma: float = 0.97,
        epsilon: float = 1.00,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.997,
        optimistic_q_init: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self.action_names = list(action_names)
        self.n_actions = len(self.action_names)

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.optimistic_q_init = float(optimistic_q_init)

        self.random_seed = int(random_seed)
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)

        self.q_table: defaultdict[str, np.ndarray] = defaultdict(
            lambda: np.full(self.n_actions, self.optimistic_q_init, dtype=float)
        )

    # --------------------------------------------------------
    # Action selection
    # --------------------------------------------------------

    def select_action(self, state_key: str, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        q_vals = self.q_table[state_key]
        max_q = np.max(q_vals)
        best_actions = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(self.rng.choice(best_actions.tolist()))

    def greedy_action(self, state_key: str) -> int:
        q_vals = self.q_table[state_key]
        max_q = np.max(q_vals)
        best_actions = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(self.rng.choice(best_actions.tolist()))

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(
        self,
        state_key: str,
        action_idx: int,
        reward: float,
        next_state_key: str,
        done: bool,
    ) -> None:
        current_q = self.q_table[state_key][action_idx]

        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state_key]))

        self.q_table[state_key][action_idx] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for state_key, q_vals in self.q_table.items():
            row = {"state_key": state_key}
            for i, action_name in enumerate(self.action_names):
                row[f"Q::{action_name}"] = float(q_vals[i])
            row["best_action"] = self.action_names[int(np.argmax(q_vals))]
            row["best_q_value"] = float(np.max(q_vals))
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["state_key", "best_action", "best_q_value"])

        return pd.DataFrame(rows).sort_values("state_key").reset_index(drop=True)

    def save_pickle(self, path: Path) -> None:
        payload = {
            "action_names": self.action_names,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "optimistic_q_init": self.optimistic_q_init,
            "random_seed": self.random_seed,
            "q_table": {k: v.tolist() for k, v in self.q_table.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_pickle(cls, path: Path) -> "QLearningAgentV2":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(
            action_names=payload["action_names"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            optimistic_q_init=payload["optimistic_q_init"],
            random_seed=payload["random_seed"],
        )

        restored = defaultdict(
            lambda: np.full(agent.n_actions, agent.optimistic_q_init, dtype=float)
        )
        for k, v in payload["q_table"].items():
            restored[k] = np.array(v, dtype=float)
        agent.q_table = restored
        return agent


# ============================================================
# Reward shaping
# ============================================================

def shape_reward(
    env_reward: float,
    next_state: dict[str, Any],
    *,
    crisis_penalty: float = 2.0,
    high_util_penalty: float = 1.0,
    blocked_penalty_scale: float = 0.0,
) -> float:
    reward = float(env_reward)

    regime = str(next_state.get("regime", "normal")).lower()
    util = safe_float(next_state.get("prev_max_utilization"), 1.0)
    blocked = safe_float(next_state.get("prev_blocked_arrivals"), 0.0)

    if regime == "crisis":
        reward -= crisis_penalty

    if util >= 1.10:
        reward -= high_util_penalty

    reward -= blocked_penalty_scale * blocked

    return reward


# ============================================================
# Episode runners
# ============================================================

def run_training_episode(
    env: HospitalRLEnv,
    agent: QLearningAgentV2,
    encoder_fn: Callable[[dict], str],
    *,
    start_regime: str,
    reward_shaping: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    state = env.reset(start_regime=start_regime)
    state_key = encoder_fn(state)

    done = False
    total_reward = 0.0
    raw_total_reward = 0.0
    step_count = 0
    actions_taken: list[str] = []

    while not done:
        action_idx = agent.select_action(state_key, explore=True)
        action_name = env.idx_to_action[action_idx]

        if verbose:
            print(
                f"  step {step_count + 1} | regime={state.get('regime')} | action={action_name}",
                flush=True,
            )

        result = env.step(action_idx)
        next_state = result.next_state
        next_state_key = encoder_fn(next_state)

        raw_reward = float(result.reward)
        final_reward = raw_reward

        if reward_shaping:
            final_reward = shape_reward(
                env_reward=raw_reward,
                next_state=next_state,
                crisis_penalty=2.0,
                high_util_penalty=1.0,
                blocked_penalty_scale=0.02,
            )

        if verbose:
            print(
                f"    raw_reward={raw_reward:.3f} | shaped_reward={final_reward:.3f} | next_regime={next_state.get('regime')}",
                flush=True,
            )

        agent.update(
            state_key=state_key,
            action_idx=action_idx,
            reward=final_reward,
            next_state_key=next_state_key,
            done=result.done,
        )

        total_reward += final_reward
        raw_total_reward += raw_reward
        step_count += 1
        actions_taken.append(action_name)

        state = next_state
        state_key = next_state_key
        done = result.done

    agent.decay_epsilon()

    return {
        "episode_reward": total_reward,
        "episode_raw_reward": raw_total_reward,
        "episode_steps": step_count,
        "final_epsilon": agent.epsilon,
        "actions_taken": actions_taken,
    }


def evaluate_policy_episode(
    env: HospitalRLEnv,
    agent: QLearningAgentV2,
    encoder_fn: Callable[[dict], str],
    *,
    start_regime: str,
) -> dict[str, Any]:
    state = env.reset(start_regime=start_regime)
    state_key = encoder_fn(state)

    done = False
    total_reward = 0.0
    step_count = 0
    actions_taken: list[str] = []
    step_rows: list[dict[str, Any]] = []

    while not done:
        action_idx = agent.greedy_action(state_key)
        action_name = env.idx_to_action[action_idx]

        result = env.step(action_idx)
        metrics = result.info.get("metrics", {})

        step_rows.append(
            {
                "timestep": env.current_step,
                "regime_before_action": result.info.get("regime_before_action"),
                "action_name": action_name,
                "reward": float(result.reward),
                **metrics,
            }
        )

        total_reward += float(result.reward)
        step_count += 1
        actions_taken.append(action_name)

        state_key = encoder_fn(result.next_state)
        done = result.done

    return {
        "episode_reward": total_reward,
        "episode_steps": step_count,
        "actions_taken": actions_taken,
        "rollout_df": pd.DataFrame(step_rows),
    }


def evaluate_fixed_policy(
    env: HospitalRLEnv,
    action_name: str,
    *,
    start_regime: str,
) -> dict[str, Any]:
    rollout_df = env.run_fixed_policy_episode(action_name)

    total_reward = 0.0
    if "reward" in rollout_df.columns:
        total_reward = float(rollout_df["reward"].sum())

    return {
        "policy_name": action_name,
        "episode_reward": total_reward,
        "episode_steps": len(rollout_df),
        "rollout_df": rollout_df,
    }


# ============================================================
# Training orchestration
# ============================================================

def train_q_learning_agent_v2(
    *,
    n_episodes: int = 500,
    start_regime_cycle: list[str] | None = None,
    use_small_encoder: bool = True,
    env_kwargs: dict | None = None,
    agent_kwargs: dict | None = None,
    verbose_every: int = 25,
) -> tuple[QLearningAgentV2, pd.DataFrame]:
    if env_kwargs is None:
        env_kwargs = {}
    if agent_kwargs is None:
        agent_kwargs = {}

    env = HospitalRLEnv(**env_kwargs)
    agent = QLearningAgentV2(
        action_names=env.get_action_space(),
        **agent_kwargs,
    )

    encoder_fn = encode_state_to_key_small if use_small_encoder else encode_state_to_key

    if start_regime_cycle is None:
        start_regime_cycle = ["normal", "surge", "crisis"]

    rows = []

    for ep in range(n_episodes):
        start_regime = start_regime_cycle[ep % len(start_regime_cycle)]
        verbose = verbose_every > 0 and ((ep + 1) % verbose_every == 0 or ep == 0)

        if verbose:
            print(
                f"\nStarting episode {ep + 1}/{n_episodes} | regime={start_regime}",
                flush=True,
            )

        result = run_training_episode(
            env=env,
            agent=agent,
            encoder_fn=encoder_fn,
            start_regime=start_regime,
            reward_shaping=True,
            verbose=verbose,
        )

        rows.append(
            {
                "episode": ep + 1,
                "start_regime": start_regime,
                "episode_reward": result["episode_reward"],
                "episode_raw_reward": result["episode_raw_reward"],
                "episode_steps": result["episode_steps"],
                "final_epsilon": result["final_epsilon"],
                "actions_taken": ",".join(result["actions_taken"]),
            }
        )

        if verbose:
            print(
                f"Completed episode {ep + 1}/{n_episodes} | "
                f"shaped_reward={result['episode_reward']:.3f} | "
                f"raw_reward={result['episode_raw_reward']:.3f} | "
                f"epsilon={agent.epsilon:.3f}",
                flush=True,
            )

    history_df = pd.DataFrame(rows)
    if not history_df.empty:
        history_df["episode_reward_rollmean_20"] = rolling_mean(
            history_df["episode_reward"].tolist(),
            20,
        )
        history_df["episode_raw_reward_rollmean_20"] = rolling_mean(
            history_df["episode_raw_reward"].tolist(),
            20,
        )

    return agent, history_df


# ============================================================
# Benchmark orchestration
# ============================================================

def benchmark_rl_vs_fixed_policies_v2(
    agent: QLearningAgentV2,
    *,
    use_small_encoder: bool = True,
    env_kwargs: dict | None = None,
    evaluation_regimes: list[str] | None = None,
    evaluation_seeds: list[int] | None = None,
    fixed_policies: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if env_kwargs is None:
        env_kwargs = {}
    if evaluation_regimes is None:
        evaluation_regimes = DEFAULT_EVAL_REGIMES.copy()
    if evaluation_seeds is None:
        evaluation_seeds = [101, 202, 303]
    if fixed_policies is None:
        fixed_policies = DEFAULT_FIXED_POLICIES.copy()

    encoder_fn = encode_state_to_key_small if use_small_encoder else encode_state_to_key

    rows = []
    rollout_store: dict[str, pd.DataFrame] = {}

    # RL controller
    for regime in evaluation_regimes:
        for seed in evaluation_seeds:
            env = HospitalRLEnv(**{**env_kwargs, "random_seed": seed})
            res = evaluate_policy_episode(
                env=env,
                agent=agent,
                encoder_fn=encoder_fn,
                start_regime=regime,
            )

            key = f"rl_policy__{regime}__seed{seed}"
            rollout_store[key] = res["rollout_df"]

            rows.append(
                {
                    "controller_type": "rl_policy",
                    "start_regime": regime,
                    "seed": seed,
                    "policy_name": "learned_q_policy",
                    "episode_reward": res["episode_reward"],
                    "episode_steps": res["episode_steps"],
                    "actions_taken": ",".join(res["actions_taken"]),
                }
            )

    # Fixed policies
    for policy_name in fixed_policies:
        for regime in evaluation_regimes:
            for seed in evaluation_seeds:
                env = HospitalRLEnv(**{**env_kwargs, "random_seed": seed})
                res = evaluate_fixed_policy(
                    env=env,
                    action_name=policy_name,
                    start_regime=regime,
                )

                key = f"{policy_name}__{regime}__seed{seed}"
                rollout_store[key] = res["rollout_df"]

                rows.append(
                    {
                        "controller_type": "fixed_policy",
                        "start_regime": regime,
                        "seed": seed,
                        "policy_name": policy_name,
                        "episode_reward": res["episode_reward"],
                        "episode_steps": res["episode_steps"],
                        "actions_taken": policy_name,
                    }
                )

    summary_df = pd.DataFrame(rows)
    return summary_df, rollout_store


# ============================================================
# Report builders
# ============================================================

def build_controller_summary(df: pd.DataFrame) -> pd.DataFrame:
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


def build_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["start_regime", "controller_type", "policy_name"], as_index=False)
        .agg(
            mean_reward=("episode_reward", "mean"),
            std_reward=("episode_reward", "std"),
            min_reward=("episode_reward", "min"),
            max_reward=("episode_reward", "max"),
            n=("episode_reward", "count"),
        )
        .sort_values(["start_regime", "mean_reward"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_headline_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in sorted(df["start_regime"].unique()):
        sub = df[df["start_regime"] == regime].copy()

        grouped = (
            sub.groupby(["controller_type", "policy_name"], as_index=False)
            .agg(mean_reward=("episode_reward", "mean"))
            .sort_values("mean_reward", ascending=False)
            .reset_index(drop=True)
        )

        best = grouped.iloc[0]
        rows.append(
            {
                "start_regime": regime,
                "best_policy": best["policy_name"],
                "controller_type": best["controller_type"],
                "best_mean_reward": best["mean_reward"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nQ-LEARNING AGENT V2 ELITE")
    print("=" * 70, flush=True)

    agent, history_df = train_q_learning_agent_v2(
        n_episodes=30,
        use_small_encoder=True,
        env_kwargs={
            "n_replications_per_step": 2,
            "max_steps": 4,
            "random_seed": 42,
        },
        agent_kwargs={
            "alpha": 0.08,
            "gamma": 0.97,
            "epsilon": 1.00,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.995,
            "optimistic_q_init": 1.0,
            "random_seed": 42,
        },
        verbose_every=50,
    )

    print("\nTraining finished.", flush=True)

    q_table_df = agent.to_dataframe()

    summary_df, rollout_store = benchmark_rl_vs_fixed_policies_v2(
        agent,
        use_small_encoder=True,
        env_kwargs={
            "n_replications_per_step": 2,
            "max_steps": 4,
        },
        evaluation_regimes=DEFAULT_EVAL_REGIMES.copy(),
        evaluation_seeds=[101, 202, 303],
        fixed_policies=DEFAULT_FIXED_POLICIES.copy(),
    )

    controller_summary = build_controller_summary(summary_df)
    regime_summary = build_regime_summary(summary_df)
    headline = build_headline_table(summary_df)

    history_path = RL_DIR / "q_learning_v2_training_history.csv"
    q_table_path = RL_DIR / "q_learning_v2_q_table.csv"
    benchmark_path = RL_DIR / "q_learning_v2_benchmark_full.csv"
    controller_path = RL_DIR / "q_learning_v2_controller_summary.csv"
    regime_path = RL_DIR / "q_learning_v2_regime_summary.csv"
    headline_path = RL_DIR / "q_learning_v2_headline.csv"
    model_path = RL_DIR / "q_learning_v2_agent.pkl"
    meta_path = RL_DIR / "q_learning_v2_run_summary.json"

    history_df.to_csv(history_path, index=False)
    q_table_df.to_csv(q_table_path, index=False)
    summary_df.to_csv(benchmark_path, index=False)
    controller_summary.to_csv(controller_path, index=False)
    regime_summary.to_csv(regime_path, index=False)
    headline.to_csv(headline_path, index=False)
    agent.save_pickle(model_path)

    for rollout_name, rollout_df in rollout_store.items():
        rollout_path = RL_DIR / f"{rollout_name}__v2_rollout.csv"
        rollout_df.to_csv(rollout_path, index=False)

    meta = {
        "n_training_episodes": int(len(history_df)),
        "n_q_states": int(len(q_table_df)),
        "n_benchmark_rows": int(len(summary_df)),
        "best_overall_policy": (
            controller_summary.iloc[0]["policy_name"] if not controller_summary.empty else None
        ),
        "best_overall_mean_reward": (
            float(controller_summary.iloc[0]["mean_reward"]) if not controller_summary.empty else None
        ),
        "final_epsilon": (
            float(history_df.iloc[-1]["final_epsilon"]) if not history_df.empty else None
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== HEADLINE WINNERS ===")
    print(headline)

    print("\n=== CONTROLLER SUMMARY ===")
    print(controller_summary)

    print("\n=== REGIME SUMMARY ===")
    print(regime_summary)

    print("\nSaved:")
    print(history_path)
    print(q_table_path)
    print(benchmark_path)
    print(controller_path)
    print(regime_path)
    print(headline_path)
    print(model_path)
    print(meta_path)


if __name__ == "__main__":
    main()