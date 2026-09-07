from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.policy_baselines import (
    build_local_only_policy_snapshot,
    build_myopic_milp_policy_snapshot,
    build_no_control_policy_snapshot,
    build_no_transfer_policy_snapshot,
    build_optimized_network_policy_snapshot,
)
from src.config.dataclasses import HealthcareInstance
from src.config.loader import load_healthcare_instance
from src.optimization.regime_robust_minco import build_regime_robust_policy_snapshot
from src.optimization.robust_minco import build_robust_optimized_network_policy_snapshot
from src.simulation.policy_runner import run_multiple_stochastic_replications


# ============================================================
# Config
# ============================================================

DEFAULT_ACTIONS = [
    "local_only",
    "myopic_milp",
    "optimized_network",
    "robust_optimized_network",
    "regime_robust_optimized_network",
]

DEFAULT_REGIME_CYCLE = [
    "normal",
    "surge",
    "crisis",
]

DEFAULT_SCENARIO_MAP = {
    "normal": "baseline",
    "surge": "network_stress",
    "crisis": "regional_crisis",
}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class EnvStepResult:
    next_state: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


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


def extract_kpi_replications(outputs: Any) -> pd.DataFrame:
    if isinstance(outputs, pd.DataFrame):
        return outputs.copy()

    if isinstance(outputs, dict):
        if "kpi_replications" in outputs:
            return outputs["kpi_replications"].copy()

        # fallback: try first dataframe-like value
        for v in outputs.values():
            if isinstance(v, pd.DataFrame):
                return v.copy()

        raise ValueError(
            f"run_multiple_stochastic_replications returned dict without "
            f"'kpi_replications'. Keys found: {list(outputs.keys())}"
        )

    raise TypeError(
        f"Unexpected output type from run_multiple_stochastic_replications: {type(outputs)}"
    )


def summarize_replication_df(rep_df: pd.DataFrame) -> dict[str, float]:
    metric_candidates = [
        "total_unsafe_excess",
        "total_blocked_arrivals",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_overflow_excess",
        "total_surge_gap",
    ]

    summary = {}
    for col in metric_candidates:
        if col in rep_df.columns:
            summary[f"{col}_mean"] = safe_float(rep_df[col].mean())
            summary[f"{col}_std"] = safe_float(rep_df[col].std(ddof=1))
        else:
            summary[f"{col}_mean"] = 0.0
            summary[f"{col}_std"] = 0.0

    return summary


# ============================================================
# Environment
# ============================================================

class HospitalRLEnv:
    """
    RL benchmark environment for meta-policy selection.

    State:
        Aggregated system indicators:
        - regime
        - previous unsafe excess
        - previous blocked arrivals
        - previous max utilization
        - previous unsafe rows
        - timestep

    Action:
        Select one policy mode from DEFAULT_ACTIONS.

    Reward:
        Negative weighted operational burden:
            -(alpha * unsafe_excess
              + beta * blocked_arrivals
              + gamma * overflow
              + delta * surge_gap)

    Transition:
        Advances one timestep and cycles / updates regime.
        In v1 this is a structured benchmark environment, not a full endogenous simulator.
    """

    def __init__(
        self,
        instance: HealthcareInstance | None = None,
        *,
        n_replications_per_step: int = 20,
        max_steps: int = 8,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.75,
        delta: float = 0.25,
        regime_cycle: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.base_instance = load_healthcare_instance() if instance is None else instance
        self.n_replications_per_step = int(n_replications_per_step)
        self.max_steps = int(max_steps)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)

        self.regime_cycle = DEFAULT_REGIME_CYCLE.copy() if regime_cycle is None else list(regime_cycle)
        self.random_seed = int(random_seed)

        self.rng = np.random.default_rng(self.random_seed)

        self.actions = DEFAULT_ACTIONS.copy()
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions)}

        self.current_step = 0
        self.current_regime = "normal"
        self.last_metrics = {
            "total_unsafe_excess_mean": 0.0,
            "total_blocked_arrivals_mean": 0.0,
            "max_utilization_ratio_mean": 1.0,
            "num_unsafe_rows_mean": 0.0,
            "total_overflow_excess_mean": 0.0,
            "total_surge_gap_mean": 0.0,
        }

    # --------------------------------------------------------
    # Core API
    # --------------------------------------------------------

    def reset(self, *, seed: int | None = None, start_regime: str = "normal") -> dict[str, Any]:
        if seed is not None:
            self.random_seed = int(seed)
            self.rng = np.random.default_rng(self.random_seed)

        self.current_step = 0
        self.current_regime = str(start_regime).strip().lower()
        self.last_metrics = {
            "total_unsafe_excess_mean": 0.0,
            "total_blocked_arrivals_mean": 0.0,
            "max_utilization_ratio_mean": 1.0,
            "num_unsafe_rows_mean": 0.0,
            "total_overflow_excess_mean": 0.0,
            "total_surge_gap_mean": 0.0,
        }

        return self.get_state()

    def step(self, action: int | str) -> EnvStepResult:
        action_name = self._resolve_action(action)

        policy_snapshot = self._build_policy_snapshot(action_name)

        outputs = run_multiple_stochastic_replications(
            instance=self.base_instance,
            policy_snapshot=policy_snapshot,
            n_replications=self.n_replications_per_step,
            base_seed=self.random_seed + 1000 * self.current_step,
        )
        rep_df = extract_kpi_replications(outputs)
        metrics = summarize_replication_df(rep_df)

        reward = self._compute_reward(metrics)

        info = {
            "action_name": action_name,
            "regime_before_action": self.current_regime,
            "metrics": metrics,
            "n_replications": self.n_replications_per_step,
        }

        self.last_metrics = metrics
        self.current_step += 1

        done = self.current_step >= self.max_steps
        if not done:
            self.current_regime = self._transition_regime()

        next_state = self.get_state()

        return EnvStepResult(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
        )

    # --------------------------------------------------------
    # State / action helpers
    # --------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        regime_code = self.regime_cycle.index(self.current_regime) if self.current_regime in self.regime_cycle else 0

        return {
            "timestep": self.current_step,
            "regime": self.current_regime,
            "regime_code": regime_code,
            "prev_unsafe_excess": self.last_metrics["total_unsafe_excess_mean"],
            "prev_blocked_arrivals": self.last_metrics["total_blocked_arrivals_mean"],
            "prev_max_utilization": self.last_metrics["max_utilization_ratio_mean"],
            "prev_unsafe_rows": self.last_metrics["num_unsafe_rows_mean"],
            "prev_overflow_excess": self.last_metrics["total_overflow_excess_mean"],
            "prev_surge_gap": self.last_metrics["total_surge_gap_mean"],
        }

    def get_action_space(self) -> list[str]:
        return self.actions.copy()

    def get_action_index(self, action_name: str) -> int:
        return self.action_to_idx[action_name]

    def _resolve_action(self, action: int | str) -> str:
        if isinstance(action, int):
            if action not in self.idx_to_action:
                raise ValueError(f"Unknown action index: {action}")
            return self.idx_to_action[action]

        action_name = str(action).strip()
        if action_name not in self.action_to_idx:
            raise ValueError(f"Unknown action name: {action_name}")

        return action_name

    # --------------------------------------------------------
    # Policy snapshot builders
    # --------------------------------------------------------

    def _build_policy_snapshot(self, action_name: str) -> dict[str, Any]:
        if action_name == "local_only":
            return build_local_only_policy_snapshot(self.base_instance)

        if action_name == "myopic_milp":
            return build_myopic_milp_policy_snapshot(self.base_instance)

        if action_name == "optimized_network":
            return build_optimized_network_policy_snapshot(self.base_instance)

        if action_name == "robust_optimized_network":
            return build_robust_optimized_network_policy_snapshot(self.base_instance)

        if action_name == "regime_robust_optimized_network":
            return build_regime_robust_policy_snapshot(
                instance=self.base_instance,
                current_regime=self.current_regime,
                lookahead_horizon=7,
                use_expected_weights=True,
            )

        raise ValueError(f"Unsupported action_name: {action_name}")

    # --------------------------------------------------------
    # Reward / transitions
    # --------------------------------------------------------

    def _compute_reward(self, metrics: dict[str, float]) -> float:
        unsafe = safe_float(metrics.get("total_unsafe_excess_mean"))
        blocked = safe_float(metrics.get("total_blocked_arrivals_mean"))
        overflow = safe_float(metrics.get("total_overflow_excess_mean"))
        surge_gap = safe_float(metrics.get("total_surge_gap_mean"))

        cost = (
            self.alpha * unsafe
            + self.beta * blocked
            + self.gamma * overflow
            + self.delta * surge_gap
        )

        return -float(cost)

    def _transition_regime(self) -> str:
        """
        Simple stochastic regime transition for v1.
        This can later be replaced by the full regime_process Markov matrix.
        """
        current = self.current_regime

        if current == "normal":
            probs = [0.70, 0.25, 0.05]
        elif current == "surge":
            probs = [0.20, 0.55, 0.25]
        else:  # crisis
            probs = [0.10, 0.30, 0.60]

        next_regime = self.rng.choice(self.regime_cycle, p=probs)
        return str(next_regime)

    # --------------------------------------------------------
    # Convenience rollout
    # --------------------------------------------------------

    def run_fixed_policy_episode(self, action_name: str) -> pd.DataFrame:
        self.reset()

        rows = []
        done = False

        while not done:
            step_result = self.step(action_name)
            row = {
                "timestep": self.current_step,
                "regime": step_result.info["regime_before_action"],
                "action_name": action_name,
                "reward": step_result.reward,
                **step_result.info["metrics"],
            }
            rows.append(row)
            done = step_result.done

        return pd.DataFrame(rows)


# ============================================================
# Example main
# ============================================================

def main() -> None:
    env = HospitalRLEnv(
        n_replications_per_step=10,
        max_steps=5,
        random_seed=42,
    )

    print("\nHOSPITAL RL ENV")
    print("=" * 60)

    state = env.reset(start_regime="surge")
    print("\nInitial state:")
    print(state)

    print("\nAvailable actions:")
    print(env.get_action_space())

    result = env.step("regime_robust_optimized_network")

    print("\nAfter one step:")
    print("Next state:", result.next_state)
    print("Reward:", result.reward)
    print("Done:", result.done)
    print("Info:", result.info)

    print("\nFixed-policy rollout example:")
    rollout_df = env.run_fixed_policy_episode("optimized_network")
    print(rollout_df)


if __name__ == "__main__":
    main()