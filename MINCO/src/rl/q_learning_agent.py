from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import pickle
import random
from typing import Callable

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
# Q-learning agent
# ============================================================

class QLearningAgent:
    def __init__(
        self,
        action_names: list[str],
        *,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 1.00,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        random_seed: int = 42,
    ) -> None:
        self.action_names = list(action_names)
        self.n_actions = len(self.action_names)

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.random_seed = int(random_seed)
        self.rng = random.Random(self.random_seed)
        np.random.seed(self.random_seed)

        self.q_table: defaultdict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
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
        return int(np.argmax(q_vals))

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
            next_max_q = float(np.max(self.q_table[next_state_key]))
            target = reward + self.gamma * next_max_q

        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state_key][action_idx] = new_q

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --------------------------------------------------------
    # Save / export
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
            return pd.DataFrame(
                columns=["state_key", "best_action", "best_q_value"]
            )

        return (
            pd.DataFrame(rows)
            .sort_values("state_key")
            .reset_index(drop=True)
        )

    def save_pickle(self, path: Path) -> None:
        payload = {
            "action_names": self.action_names,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "random_seed": self.random_seed,
            "q_table": {k: v.tolist() for k, v in self.q_table.items()},
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_pickle(cls, path: Path) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(
            action_names=payload["action_names"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            random_seed=payload["random_seed"],
        )

        restored = defaultdict(
            lambda: np.zeros(agent.n_actions, dtype=float)
        )

        for k, v in payload["q_table"].items():
            restored[k] = np.array(v, dtype=float)

        agent.q_table = restored
        return agent


# ============================================================
# Training helpers
# ============================================================

def run_training_episode(
    env: HospitalRLEnv,
    agent: QLearningAgent,
    encoder_fn: Callable[[dict], str],
    *,
    start_regime: str = "normal",
) -> dict:

    state = env.reset(start_regime=start_regime)
    state_key = encoder_fn(state)

    done = False
    total_reward = 0.0
    step_count = 0
    actions_taken: list[str] = []

    while not done:
        action_idx = agent.select_action(state_key, explore=True)
        action_name = env.idx_to_action[action_idx]

        print(
            f"  step {step_count + 1} | regime={state['regime']} | action={action_name}",
            flush=True,
        )

        result = env.step(action_idx)

        next_state = result.next_state
        next_state_key = encoder_fn(next_state)

        print(
            f"    reward={result.reward:.3f} | next_regime={next_state['regime']}",
            flush=True,
        )

        agent.update(
            state_key=state_key,
            action_idx=action_idx,
            reward=result.reward,
            next_state_key=next_state_key,
            done=result.done,
        )

        total_reward += float(result.reward)
        step_count += 1
        actions_taken.append(action_name)

        state = next_state
        state_key = next_state_key
        done = result.done

    agent.decay_epsilon()

    return {
        "episode_reward": total_reward,
        "episode_steps": step_count,
        "final_epsilon": agent.epsilon,
        "actions_taken": actions_taken,
    }


def evaluate_policy_episode(
    env: HospitalRLEnv,
    agent: QLearningAgent,
    encoder_fn: Callable[[dict], str],
    *,
    start_regime: str = "normal",
) -> dict:

    state = env.reset(start_regime=start_regime)
    state_key = encoder_fn(state)

    done = False
    total_reward = 0.0
    step_count = 0
    actions_taken: list[str] = []
    step_rows: list[dict] = []

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
    start_regime: str = "normal",
) -> dict:

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
# Experiment runners
# ============================================================

def train_q_learning_agent(
    *,
    n_episodes: int = 5,
    start_regime_cycle: list[str] | None = None,
    use_small_encoder: bool = True,
    env_kwargs: dict | None = None,
    agent_kwargs: dict | None = None,
) -> tuple[QLearningAgent, pd.DataFrame]:

    if env_kwargs is None:
        env_kwargs = {}

    if agent_kwargs is None:
        agent_kwargs = {}

    env = HospitalRLEnv(**env_kwargs)

    agent = QLearningAgent(
        action_names=env.get_action_space(),
        **agent_kwargs,
    )

    encoder_fn = (
        encode_state_to_key_small
        if use_small_encoder
        else encode_state_to_key
    )

    if start_regime_cycle is None:
        start_regime_cycle = ["normal", "surge", "crisis"]

    rows = []

    for ep in range(n_episodes):
        start_regime = start_regime_cycle[ep % len(start_regime_cycle)]

        print(
            f"\nStarting episode {ep + 1}/{n_episodes} | regime={start_regime}",
            flush=True,
        )

        result = run_training_episode(
            env=env,
            agent=agent,
            encoder_fn=encoder_fn,
            start_regime=start_regime,
        )

        rows.append(
            {
                "episode": ep + 1,
                "start_regime": start_regime,
                "episode_reward": result["episode_reward"],
                "episode_steps": result["episode_steps"],
                "final_epsilon": result["final_epsilon"],
                "actions_taken": ",".join(result["actions_taken"]),
            }
        )

        print(
            f"Completed episode {ep + 1}/{n_episodes} | "
            f"reward={result['episode_reward']:.3f} | "
            f"epsilon={agent.epsilon:.3f}",
            flush=True,
        )

    history_df = pd.DataFrame(rows)
    return agent, history_df


def DEFAULT_FIXED_POLICY_SET() -> list[str]:
    return [
        "local_only",
        "myopic_milp",
        "optimized_network",
        "robust_optimized_network",
        "regime_robust_optimized_network",
    ]


def benchmark_rl_vs_fixed_policies(
    agent: QLearningAgent,
    *,
    use_small_encoder: bool = True,
    env_kwargs: dict | None = None,
    evaluation_regimes: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:

    if env_kwargs is None:
        env_kwargs = {}

    if evaluation_regimes is None:
        evaluation_regimes = ["normal"]

    encoder_fn = (
        encode_state_to_key_small
        if use_small_encoder
        else encode_state_to_key
    )

    rows = []
    rollout_store: dict[str, pd.DataFrame] = {}

    # RL controller
    for regime in evaluation_regimes:
        env = HospitalRLEnv(**env_kwargs)

        res = evaluate_policy_episode(
            env=env,
            agent=agent,
            encoder_fn=encoder_fn,
            start_regime=regime,
        )

        rollout_store[f"rl_policy__{regime}"] = res["rollout_df"]

        rows.append(
            {
                "controller_type": "rl_policy",
                "start_regime": regime,
                "policy_name": "learned_q_policy",
                "episode_reward": res["episode_reward"],
                "episode_steps": res["episode_steps"],
                "actions_taken": ",".join(res["actions_taken"]),
            }
        )

    # Fixed policies
    for policy_name in DEFAULT_FIXED_POLICY_SET():
        for regime in evaluation_regimes:
            env = HospitalRLEnv(**env_kwargs)

            res = evaluate_fixed_policy(
                env=env,
                action_name=policy_name,
                start_regime=regime,
            )

            rollout_store[f"{policy_name}__{regime}"] = res["rollout_df"]

            rows.append(
                {
                    "controller_type": "fixed_policy",
                    "start_regime": regime,
                    "policy_name": policy_name,
                    "episode_reward": res["episode_reward"],
                    "episode_steps": res["episode_steps"],
                    "actions_taken": policy_name,
                }
            )

    summary_df = (
        pd.DataFrame(rows)
        .sort_values(
            ["start_regime", "episode_reward"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return summary_df, rollout_store


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nQ-LEARNING AGENT")
    print("=" * 60, flush=True)

    agent, history_df = train_q_learning_agent(
        n_episodes=5,
        use_small_encoder=True,
        env_kwargs={
            "n_replications_per_step": 1,
            "max_steps": 2,
            "random_seed": 42,
        },
        agent_kwargs={
            "alpha": 0.10,
            "gamma": 0.95,
            "epsilon": 1.00,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.95,
            "random_seed": 42,
        },
    )

    print("\nTraining finished.", flush=True)

    q_table_df = agent.to_dataframe()

    summary_df, rollout_store = benchmark_rl_vs_fixed_policies(
        agent,
        use_small_encoder=True,
        env_kwargs={
            "n_replications_per_step": 1,
            "max_steps": 2,
            "random_seed": 99,
        },
        evaluation_regimes=["normal"],
    )

    history_path = RL_DIR / "q_learning_training_history.csv"
    q_table_path = RL_DIR / "q_learning_q_table.csv"
    summary_path = RL_DIR / "q_learning_benchmark_summary.csv"
    model_path = RL_DIR / "q_learning_agent.pkl"
    meta_path = RL_DIR / "q_learning_run_summary.json"

    history_df.to_csv(history_path, index=False)
    q_table_df.to_csv(q_table_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    agent.save_pickle(model_path)

    for rollout_name, rollout_df in rollout_store.items():
        rollout_path = RL_DIR / f"{rollout_name}__rollout.csv"
        rollout_df.to_csv(rollout_path, index=False)

    meta = {
        "n_training_episodes": int(len(history_df)),
        "n_q_states": int(len(q_table_df)),
        "n_benchmark_rows": int(len(summary_df)),
        "best_benchmark_row": (
            summary_df.iloc[0].to_dict()
            if not summary_df.empty
            else {}
        ),
    }

    meta_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("\nTraining history:")
    print(history_df)

    print("\nQ-table head:")
    print(q_table_df.head())

    print("\nBenchmark summary:")
    print(summary_df)

    print("\nSaved:")
    print(history_path)
    print(q_table_path)
    print(summary_path)
    print(model_path)
    print(meta_path)


if __name__ == "__main__":
    main()