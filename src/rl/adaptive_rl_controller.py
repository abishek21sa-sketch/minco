from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd

from src.rl.hospital_env import HospitalRLEnv
from src.rl.state_encoder import encode_state_to_key_small


RESULTS_DIR = Path("results")
RL_DIR = RESULTS_DIR / "rl"
ADAPTIVE_RL_DIR = RL_DIR / "adaptive_robust_rl"
ADAPTIVE_RL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Adaptive robust action modes
# ============================================================

ADAPTIVE_ACTIONS = {
    "access_preserving": {
        "base_policy": "optimized_network",
        "description": "Prioritize access and throughput under stable conditions.",
        "safety_weight": 0.85,
        "access_weight": 0.45,
        "overflow_weight": 0.70,
        "surge_weight": 0.25,
        "crisis_penalty": 1.0,
        "util_penalty": 0.5,
    },
    "balanced_robust": {
        "base_policy": "robust_optimized_network",
        "description": "Balance safety protection and access preservation.",
        "safety_weight": 1.00,
        "access_weight": 0.55,
        "overflow_weight": 0.80,
        "surge_weight": 0.30,
        "crisis_penalty": 1.5,
        "util_penalty": 0.75,
    },
    "safety_robust": {
        "base_policy": "robust_optimized_network",
        "description": "Increase safety protection during overload risk.",
        "safety_weight": 1.35,
        "access_weight": 0.45,
        "overflow_weight": 1.00,
        "surge_weight": 0.35,
        "crisis_penalty": 2.0,
        "util_penalty": 1.0,
    },
    "crisis_regime_robust": {
        "base_policy": "regime_robust_optimized_network",
        "description": "Use regime-aware protection under crisis-like dynamics.",
        "safety_weight": 1.60,
        "access_weight": 0.35,
        "overflow_weight": 1.15,
        "surge_weight": 0.45,
        "crisis_penalty": 2.5,
        "util_penalty": 1.25,
    },
    "access_recovery": {
        "base_policy": "optimized_network",
        "description": "Relax conservatism after blocked-arrival pressure.",
        "safety_weight": 0.90,
        "access_weight": 0.75,
        "overflow_weight": 0.75,
        "surge_weight": 0.25,
        "crisis_penalty": 1.0,
        "util_penalty": 0.5,
    },
}


ACTION_NAMES = list(ADAPTIVE_ACTIONS.keys())


# ============================================================
# Data containers
# ============================================================

@dataclass
class AdaptiveStepResult:
    next_state: dict[str, Any]
    raw_reward: float
    shaped_reward: float
    done: bool
    info: dict[str, Any]


# ============================================================
# Helpers
# ============================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
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


def adaptive_reward_from_metrics(
    metrics: dict[str, Any],
    next_state: dict[str, Any],
    action_mode: str,
) -> float:
    cfg = ADAPTIVE_ACTIONS[action_mode]

    unsafe = safe_float(metrics.get("total_unsafe_excess_mean"))
    blocked = safe_float(metrics.get("total_blocked_arrivals_mean"))
    overflow = safe_float(metrics.get("total_overflow_excess_mean"))
    surge_gap = safe_float(metrics.get("total_surge_gap_mean"))
    util = safe_float(metrics.get("max_utilization_ratio_mean"), 1.0)

    cost = (
        cfg["safety_weight"] * unsafe
        + cfg["access_weight"] * blocked
        + cfg["overflow_weight"] * overflow
        + cfg["surge_weight"] * surge_gap
    )

    regime = str(next_state.get("regime", "normal")).lower()

    if regime == "crisis":
        cost += cfg["crisis_penalty"]

    if util >= 1.10:
        cost += cfg["util_penalty"]

    # penalty if access-preserving mode is used while unsafe risk is already high
    if action_mode in {"access_preserving", "access_recovery"} and unsafe >= 10:
        cost += 2.0

    # penalty if crisis mode is overused in normal conditions with blocked arrivals high
    if action_mode == "crisis_regime_robust" and regime == "normal" and blocked >= 10:
        cost += 1.5

    return -float(cost)


def summarize_action_usage(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in history_df.iterrows():
        actions = str(row.get("actions_taken", "")).split(",")
        actions = [a for a in actions if a]

        for a in actions:
            rows.append(
                {
                    "episode": row["episode"],
                    "config": row.get("config", "default"),
                    "action_mode": a,
                }
            )

    if not rows:
        return pd.DataFrame()

    usage = pd.DataFrame(rows)

    out = (
        usage.groupby(["config", "action_mode"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    out["share"] = out["count"] / out.groupby("config")["count"].transform("sum")
    return out.sort_values(["config", "share"], ascending=[True, False]).reset_index(drop=True)


# ============================================================
# Adaptive Q-agent
# ============================================================

class AdaptiveRobustQLearningAgent:
    def __init__(
        self,
        action_names: list[str] | None = None,
        *,
        alpha: float = 0.08,
        gamma: float = 0.97,
        epsilon: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.995,
        optimistic_q_init: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.action_names = ACTION_NAMES.copy() if action_names is None else list(action_names)
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

    def select_action(self, state_key: str, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        q_vals = self.q_table[state_key]
        max_q = np.max(q_vals)
        best = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(self.rng.choice(best.tolist()))

    def greedy_action(self, state_key: str) -> int:
        q_vals = self.q_table[state_key]
        max_q = np.max(q_vals)
        best = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(self.rng.choice(best.tolist()))

    def update(
        self,
        state_key: str,
        action_idx: int,
        reward: float,
        next_state_key: str,
        done: bool,
    ) -> None:
        old_q = self.q_table[state_key][action_idx]

        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state_key]))

        self.q_table[state_key][action_idx] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []

        for state_key, q_vals in self.q_table.items():
            row = {"state_key": state_key}
            for i, action_name in enumerate(self.action_names):
                row[f"Q::{action_name}"] = float(q_vals[i])

            best_idx = int(np.argmax(q_vals))
            row["best_action_mode"] = self.action_names[best_idx]
            row["best_base_policy"] = ADAPTIVE_ACTIONS[self.action_names[best_idx]]["base_policy"]
            row["best_q_value"] = float(q_vals[best_idx])
            rows.append(row)

        if not rows:
            return pd.DataFrame()

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
    def load_pickle(cls, path: Path) -> "AdaptiveRobustQLearningAgent":
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
# Adaptive controller wrapper
# ============================================================

class AdaptiveRobustRLEnvWrapper:
    """
    Wrapper around HospitalRLEnv.

    RL chooses adaptive robust action modes.
    Each action mode maps to a base policy already supported by HospitalRLEnv.

    This is v1 of RL-guided robust optimization:
    - RL controls when to use nominal, robust, or regime-robust decisions.
    - Reward shaping differs by action mode.
    - Later versions can pass gamma/access/safety parameters directly into MILP.
    """

    def __init__(
        self,
        *,
        n_replications_per_step: int = 1,
        max_steps: int = 4,
        random_seed: int = 42,
    ) -> None:
        self.env = HospitalRLEnv(
            n_replications_per_step=n_replications_per_step,
            max_steps=max_steps,
            random_seed=random_seed,
        )
        self.action_names = ACTION_NAMES.copy()
        self.action_to_idx = {a: i for i, a in enumerate(self.action_names)}
        self.idx_to_action = {i: a for i, a in enumerate(self.action_names)}

    def reset(self, *, start_regime: str = "normal") -> dict[str, Any]:
        return self.env.reset(start_regime=start_regime)

    def step(self, adaptive_action_idx: int | str) -> AdaptiveStepResult:
        action_mode = self._resolve_action(adaptive_action_idx)
        base_policy = ADAPTIVE_ACTIONS[action_mode]["base_policy"]

        raw_result = self.env.step(base_policy)

        metrics = raw_result.info.get("metrics", {})
        shaped_reward = adaptive_reward_from_metrics(
            metrics=metrics,
            next_state=raw_result.next_state,
            action_mode=action_mode,
        )

        info = dict(raw_result.info)
        info["adaptive_action_mode"] = action_mode
        info["base_policy"] = base_policy
        info["action_description"] = ADAPTIVE_ACTIONS[action_mode]["description"]
        info["raw_reward"] = raw_result.reward
        info["shaped_reward"] = shaped_reward

        return AdaptiveStepResult(
            next_state=raw_result.next_state,
            raw_reward=float(raw_result.reward),
            shaped_reward=float(shaped_reward),
            done=raw_result.done,
            info=info,
        )

    def _resolve_action(self, action: int | str) -> str:
        if isinstance(action, int):
            if action not in self.idx_to_action:
                raise ValueError(f"Unknown action index: {action}")
            return self.idx_to_action[action]

        action_name = str(action)
        if action_name not in self.action_to_idx:
            raise ValueError(f"Unknown adaptive action: {action_name}")

        return action_name

    def get_action_space(self) -> list[str]:
        return self.action_names.copy()


# ============================================================
# Training / evaluation
# ============================================================

def run_training_episode(
    env: AdaptiveRobustRLEnvWrapper,
    agent: AdaptiveRobustQLearningAgent,
    *,
    start_regime: str,
    verbose: bool = False,
) -> dict[str, Any]:
    state = env.reset(start_regime=start_regime)
    state_key = encode_state_to_key_small(state)

    done = False
    total_raw_reward = 0.0
    total_shaped_reward = 0.0
    step_count = 0
    action_modes = []
    base_policies = []

    while not done:
        action_idx = agent.select_action(state_key, explore=True)
        action_mode = agent.action_names[action_idx]

        if verbose:
            print(
                f"  step {step_count + 1} | regime={state.get('regime')} | adaptive_action={action_mode}",
                flush=True,
            )

        result = env.step(action_idx)

        next_state_key = encode_state_to_key_small(result.next_state)

        agent.update(
            state_key=state_key,
            action_idx=action_idx,
            reward=result.shaped_reward,
            next_state_key=next_state_key,
            done=result.done,
        )

        total_raw_reward += result.raw_reward
        total_shaped_reward += result.shaped_reward
        step_count += 1
        action_modes.append(action_mode)
        base_policies.append(str(result.info.get("base_policy")))

        state = result.next_state
        state_key = next_state_key
        done = result.done

        if verbose:
            print(
                f"    raw={result.raw_reward:.3f} | shaped={result.shaped_reward:.3f} | "
                f"base_policy={result.info.get('base_policy')} | next_regime={state.get('regime')}",
                flush=True,
            )

    agent.decay_epsilon()

    return {
        "episode_raw_reward": total_raw_reward,
        "episode_shaped_reward": total_shaped_reward,
        "episode_steps": step_count,
        "final_epsilon": agent.epsilon,
        "actions_taken": action_modes,
        "base_policies_used": base_policies,
    }


def train_adaptive_controller(
    *,
    n_episodes: int = 100,
    n_replications_per_step: int = 1,
    max_steps: int = 4,
    start_regime_cycle: list[str] | None = None,
    random_seed: int = 42,
    verbose_every: int = 20,
) -> tuple[AdaptiveRobustQLearningAgent, pd.DataFrame]:
    if start_regime_cycle is None:
        start_regime_cycle = ["normal", "surge", "crisis"]

    env = AdaptiveRobustRLEnvWrapper(
        n_replications_per_step=n_replications_per_step,
        max_steps=max_steps,
        random_seed=random_seed,
    )

    agent = AdaptiveRobustQLearningAgent(
        action_names=env.get_action_space(),
        alpha=0.08,
        gamma=0.97,
        epsilon=1.0,
        epsilon_min=0.03,
        epsilon_decay=0.995,
        optimistic_q_init=1.0,
        random_seed=random_seed,
    )

    rows = []

    for ep in range(n_episodes):
        start_regime = start_regime_cycle[ep % len(start_regime_cycle)]
        verbose = ep == 0 or ((ep + 1) % verbose_every == 0)

        if verbose:
            print(
                f"\nAdaptive RL episode {ep + 1}/{n_episodes} | start_regime={start_regime}",
                flush=True,
            )

        result = run_training_episode(
            env=env,
            agent=agent,
            start_regime=start_regime,
            verbose=verbose,
        )

        rows.append(
            {
                "episode": ep + 1,
                "config": "adaptive_robust_rl",
                "start_regime": start_regime,
                "episode_raw_reward": result["episode_raw_reward"],
                "episode_shaped_reward": result["episode_shaped_reward"],
                "episode_steps": result["episode_steps"],
                "final_epsilon": result["final_epsilon"],
                "actions_taken": ",".join(result["actions_taken"]),
                "base_policies_used": ",".join(result["base_policies_used"]),
            }
        )

        if verbose:
            print(
                f"Completed episode {ep + 1}/{n_episodes} | "
                f"raw={result['episode_raw_reward']:.3f} | "
                f"shaped={result['episode_shaped_reward']:.3f} | "
                f"eps={agent.epsilon:.3f}",
                flush=True,
            )

    history_df = pd.DataFrame(rows)

    if not history_df.empty:
        history_df["raw_reward_rollmean_20"] = rolling_mean(
            history_df["episode_raw_reward"].tolist(),
            20,
        )
        history_df["shaped_reward_rollmean_20"] = rolling_mean(
            history_df["episode_shaped_reward"].tolist(),
            20,
        )

    return agent, history_df


def evaluate_adaptive_controller(
    agent: AdaptiveRobustQLearningAgent,
    *,
    start_regimes: list[str],
    seeds: list[int],
    n_replications_per_step: int = 1,
    max_steps: int = 4,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    rollout_store = {}

    for regime in start_regimes:
        for seed in seeds:
            env = AdaptiveRobustRLEnvWrapper(
                n_replications_per_step=n_replications_per_step,
                max_steps=max_steps,
                random_seed=seed,
            )

            state = env.reset(start_regime=regime)
            state_key = encode_state_to_key_small(state)

            done = False
            total_raw_reward = 0.0
            total_shaped_reward = 0.0
            actions_taken = []
            base_policies = []
            step_rows = []
            step_count = 0

            while not done:
                action_idx = agent.greedy_action(state_key)
                action_mode = agent.action_names[action_idx]

                result = env.step(action_idx)

                total_raw_reward += result.raw_reward
                total_shaped_reward += result.shaped_reward
                actions_taken.append(action_mode)
                base_policies.append(str(result.info.get("base_policy")))

                metrics = result.info.get("metrics", {})

                step_rows.append(
                    {
                        "step": step_count + 1,
                        "start_regime": regime,
                        "seed": seed,
                        "adaptive_action_mode": action_mode,
                        "base_policy": result.info.get("base_policy"),
                        "raw_reward": result.raw_reward,
                        "shaped_reward": result.shaped_reward,
                        "regime_before_action": result.info.get("regime_before_action"),
                        **metrics,
                    }
                )

                state_key = encode_state_to_key_small(result.next_state)
                done = result.done
                step_count += 1

            rows.append(
                {
                    "controller_type": "adaptive_rl_policy",
                    "policy_name": "adaptive_robust_q_policy",
                    "start_regime": regime,
                    "seed": seed,
                    "episode_raw_reward": total_raw_reward,
                    "episode_shaped_reward": total_shaped_reward,
                    "episode_steps": step_count,
                    "actions_taken": ",".join(actions_taken),
                    "base_policies_used": ",".join(base_policies),
                }
            )

            rollout_store[f"adaptive_rl__{regime}__seed{seed}"] = pd.DataFrame(step_rows)

    return pd.DataFrame(rows), rollout_store


def evaluate_fixed_policy(
    *,
    policy_name: str,
    start_regimes: list[str],
    seeds: list[int],
    n_replications_per_step: int = 1,
    max_steps: int = 4,
) -> pd.DataFrame:
    rows = []

    for regime in start_regimes:
        for seed in seeds:
            env = HospitalRLEnv(
                n_replications_per_step=n_replications_per_step,
                max_steps=max_steps,
                random_seed=seed,
            )

            rollout_df = env.run_fixed_policy_episode(policy_name)

            raw_reward = (
                float(rollout_df["reward"].sum())
                if "reward" in rollout_df.columns
                else 0.0
            )

            rows.append(
                {
                    "controller_type": "fixed_policy",
                    "policy_name": policy_name,
                    "start_regime": regime,
                    "seed": seed,
                    "episode_raw_reward": raw_reward,
                    "episode_shaped_reward": raw_reward,
                    "episode_steps": len(rollout_df),
                    "actions_taken": policy_name,
                    "base_policies_used": policy_name,
                }
            )

    return pd.DataFrame(rows)


def summarize_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()

    return (
        eval_df.groupby(["controller_type", "policy_name"], as_index=False)
        .agg(
            mean_raw_reward=("episode_raw_reward", "mean"),
            std_raw_reward=("episode_raw_reward", "std"),
            mean_shaped_reward=("episode_shaped_reward", "mean"),
            std_shaped_reward=("episode_shaped_reward", "std"),
            min_raw_reward=("episode_raw_reward", "min"),
            max_raw_reward=("episode_raw_reward", "max"),
            n=("episode_raw_reward", "count"),
        )
        .sort_values("mean_raw_reward", ascending=False)
        .reset_index(drop=True)
    )


def build_headline(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    ranked = summary_df.sort_values("mean_raw_reward", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    best = ranked.iloc[0]
    adaptive = ranked[ranked["policy_name"] == "adaptive_robust_q_policy"]

    if adaptive.empty:
        adaptive_rank = None
        adaptive_reward = None
        gap = None
    else:
        ar = adaptive.iloc[0]
        adaptive_rank = int(ar["rank"])
        adaptive_reward = float(ar["mean_raw_reward"])
        gap = adaptive_reward - float(best["mean_raw_reward"])

    return pd.DataFrame(
        [
            {
                "best_policy": best["policy_name"],
                "best_controller_type": best["controller_type"],
                "best_mean_raw_reward": float(best["mean_raw_reward"]),
                "adaptive_rl_rank": adaptive_rank,
                "adaptive_rl_mean_raw_reward": adaptive_reward,
                "gap_adaptive_rl_minus_best": gap,
                "n_controllers": len(ranked),
            }
        ]
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nADAPTIVE ROBUST RL CONTROLLER")
    print("=" * 70, flush=True)

    agent, history_df = train_adaptive_controller(
        n_episodes=80,
        n_replications_per_step=1,
        max_steps=4,
        start_regime_cycle=["normal", "surge", "crisis"],
        random_seed=42,
        verbose_every=20,
    )

    q_table_df = agent.to_dataframe()
    action_usage_df = summarize_action_usage(history_df)

    eval_regimes = ["normal", "surge", "crisis"]
    eval_seeds = [101, 202, 303]

    adaptive_eval_df, rollout_store = evaluate_adaptive_controller(
        agent,
        start_regimes=eval_regimes,
        seeds=eval_seeds,
        n_replications_per_step=1,
        max_steps=4,
    )

    fixed_frames = []
    for policy in [
        "optimized_network",
        "robust_optimized_network",
        "regime_robust_optimized_network",
        "local_only",
        "myopic_milp",
    ]:
        fixed_frames.append(
            evaluate_fixed_policy(
                policy_name=policy,
                start_regimes=eval_regimes,
                seeds=eval_seeds,
                n_replications_per_step=1,
                max_steps=4,
            )
        )

    fixed_eval_df = pd.concat(fixed_frames, ignore_index=True)
    combined_eval_df = pd.concat([adaptive_eval_df, fixed_eval_df], ignore_index=True)

    summary_df = summarize_eval(combined_eval_df)
    headline_df = build_headline(summary_df)

    history_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_training_history.csv"
    qtable_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_q_table.csv"
    usage_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_action_usage.csv"
    eval_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_eval.csv"
    summary_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_summary.csv"
    headline_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_headline.csv"
    model_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_agent.pkl"
    meta_path = ADAPTIVE_RL_DIR / "adaptive_robust_rl_report.json"

    history_df.to_csv(history_path, index=False)
    q_table_df.to_csv(qtable_path, index=False)
    action_usage_df.to_csv(usage_path, index=False)
    combined_eval_df.to_csv(eval_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    headline_df.to_csv(headline_path, index=False)
    agent.save_pickle(model_path)

    for name, df in rollout_store.items():
        df.to_csv(ADAPTIVE_RL_DIR / f"{name}_rollout.csv", index=False)

    report = {
        "n_training_episodes": int(len(history_df)),
        "n_q_states": int(len(q_table_df)),
        "n_eval_rows": int(len(combined_eval_df)),
        "headline": headline_df.iloc[0].to_dict() if not headline_df.empty else {},
        "adaptive_action_modes": ADAPTIVE_ACTIONS,
    }
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Headline ===")
    print(headline_df.to_string(index=False))

    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Action Usage ===")
    print(action_usage_df.to_string(index=False))

    print("\nSaved:")
    print(history_path)
    print(qtable_path)
    print(usage_path)
    print(eval_path)
    print(summary_path)
    print(headline_path)
    print(model_path)
    print(meta_path)


if __name__ == "__main__":
    main()