from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd

from src.rl.hospital_env import HospitalRLEnv
from src.rl.state_encoder import encode_state_to_key_small


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "rl" / "adaptive_rl_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLEAN ACTION SPACE (VERY IMPORTANT)
# ============================================================

ACTIONS = {
    "access_preserving": "optimized_network",
    "balanced_robust": "robust_optimized_network",
    "crisis_regime_robust": "regime_robust_optimized_network",
}

ACTION_NAMES = list(ACTIONS.keys())


# ============================================================
# AGENT
# ============================================================

class AdaptiveRLV2Agent:
    def __init__(
        self,
        *,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.96,
        optimistic_q: float = 1.0,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.n_actions = len(ACTION_NAMES)

        self.q = defaultdict(
            lambda: np.full(self.n_actions, optimistic_q, dtype=float)
        )

    def select(self, state_key: str, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        q_vals = self.q[state_key]
        max_q = np.max(q_vals)
        best = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(self.rng.choice(best.tolist()))

    def update(self, s, a, r, s_next, done):
        old = self.q[s][a]

        if done:
            target = r
        else:
            target = r + self.gamma * float(np.max(self.q[s_next]))

        self.q[s][a] = old + self.alpha * (target - old)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy(self, state_key):
        q_vals = self.q[state_key]
        return int(np.argmax(q_vals))


# ============================================================
# TRAIN
# ============================================================

def train_agent():
    env = HospitalRLEnv(
        n_replications_per_step=1,
        max_steps=4,
        random_seed=42,
    )

    agent = AdaptiveRLV2Agent()

    rows = []

    regimes = ["normal", "surge", "crisis"]

    for ep in range(200):

        start_regime = regimes[ep % len(regimes)]

        state = env.reset(start_regime=start_regime)
        s_key = encode_state_to_key_small(state)

        done = False
        total_reward = 0.0
        actions = []

        while not done:
            a_idx = agent.select(s_key, explore=True)
            action_name = ACTION_NAMES[a_idx]
            base_policy = ACTIONS[action_name]

            result = env.step(base_policy)

            r = float(result.reward)   # <-- RAW REWARD ONLY
            s_next = encode_state_to_key_small(result.next_state)

            agent.update(s_key, a_idx, r, s_next, result.done)

            total_reward += r
            actions.append(action_name)

            s_key = s_next
            done = result.done

        agent.decay()

        rows.append({
            "episode": ep + 1,
            "start_regime": start_regime,
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "actions": ",".join(actions),
        })

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"ep {ep+1}/200 | reward={total_reward:.3f} | eps={agent.epsilon:.3f}")

    df = pd.DataFrame(rows)
    df["roll_mean_20"] = df["reward"].rolling(20, min_periods=1).mean()

    return agent, df


# ============================================================
# EVALUATE
# ============================================================

def evaluate(agent):
    regimes = ["normal", "surge", "crisis"]
    seeds = [101, 202, 303]

    rows = []

    for r in regimes:
        for seed in seeds:
            env = HospitalRLEnv(
                n_replications_per_step=1,
                max_steps=4,
                random_seed=seed,
            )

            state = env.reset(start_regime=r)
            s_key = encode_state_to_key_small(state)

            total = 0.0
            done = False

            while not done:
                a_idx = agent.greedy(s_key)
                action_name = ACTION_NAMES[a_idx]
                base_policy = ACTIONS[action_name]

                result = env.step(base_policy)

                total += float(result.reward)
                s_key = encode_state_to_key_small(result.next_state)
                done = result.done

            rows.append({
                "controller": "adaptive_rl_v2",
                "start_regime": r,
                "seed": seed,
                "reward": total,
            })

    return pd.DataFrame(rows)


def evaluate_fixed():
    regimes = ["normal", "surge", "crisis"]
    seeds = [101, 202, 303]

    policies = [
        "optimized_network",
        "robust_optimized_network",
        "regime_robust_optimized_network",
        "local_only",
        "myopic_milp",
    ]

    rows = []

    for p in policies:
        for r in regimes:
            for seed in seeds:
                env = HospitalRLEnv(
                    n_replications_per_step=1,
                    max_steps=4,
                    random_seed=seed,
                )

                rollout = env.run_fixed_policy_episode(p)

                total = float(rollout["reward"].sum())

                rows.append({
                    "controller": "fixed",
                    "policy": p,
                    "start_regime": r,
                    "seed": seed,
                    "reward": total,
                })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nADAPTIVE RL V2\n" + "="*60)

    agent, train_df = train_agent()

    rl_df = evaluate(agent)
    fx_df = evaluate_fixed()

    fx_summary = fx_df.groupby("policy", as_index=False)["reward"].mean()
    best_fx = fx_summary.sort_values("reward", ascending=False).iloc[0]

    rl_mean = rl_df["reward"].mean()

    print("\n=== RESULT ===")
    print(f"Best fixed: {best_fx['policy']} = {best_fx['reward']:.3f}")
    print(f"RL v2: {rl_mean:.3f}")
    print(f"Gap: {rl_mean - best_fx['reward']:.3f}")

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    rl_df.to_csv(OUT_DIR / "eval_rl.csv", index=False)
    fx_df.to_csv(OUT_DIR / "eval_fixed.csv", index=False)

    print("\nSaved → results/rl/adaptive_rl_v2/")


if __name__ == "__main__":
    main()