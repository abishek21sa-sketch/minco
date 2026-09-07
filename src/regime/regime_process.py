from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_REGIMES = ["normal", "surge", "crisis"]


@dataclass(frozen=True)
class RegimeProcess:
    regimes: list[str]
    transition_matrix: pd.DataFrame

    def validate(self) -> None:
        if self.transition_matrix.empty:
            raise ValueError("Transition matrix is empty.")

        missing_rows = [r for r in self.regimes if r not in self.transition_matrix.index]
        missing_cols = [c for c in self.regimes if c not in self.transition_matrix.columns]

        if missing_rows:
            raise ValueError(f"Missing transition-matrix rows for regimes: {missing_rows}")
        if missing_cols:
            raise ValueError(f"Missing transition-matrix cols for regimes: {missing_cols}")

        tm = self.transition_matrix.loc[self.regimes, self.regimes].astype(float)

        if (tm.values < 0).any():
            raise ValueError("Transition matrix contains negative probabilities.")

        row_sums = tm.sum(axis=1).to_numpy()
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(
                f"Transition matrix rows must sum to 1. Row sums found: {row_sums.tolist()}"
            )

    def transition_prob(self, current_regime: str, next_regime: str) -> float:
        self.validate()
        return float(self.transition_matrix.loc[current_regime, next_regime])

    def row(self, current_regime: str) -> pd.Series:
        self.validate()
        return self.transition_matrix.loc[current_regime, self.regimes].astype(float).copy()


def build_default_transition_matrix() -> pd.DataFrame:
    """
    Default regime transition matrix for:
    normal -> surge -> crisis

    Interpretation:
    - normal tends to persist, but can move into surge
    - surge can recover, persist, or worsen
    - crisis is sticky, but may partially recover
    """
    tm = pd.DataFrame(
        data=[
            [0.75, 0.20, 0.05],  # from normal
            [0.25, 0.55, 0.20],  # from surge
            [0.10, 0.35, 0.55],  # from crisis
        ],
        index=DEFAULT_REGIMES,
        columns=DEFAULT_REGIMES,
    )
    return tm


def build_default_regime_process() -> RegimeProcess:
    rp = RegimeProcess(
        regimes=DEFAULT_REGIMES.copy(),
        transition_matrix=build_default_transition_matrix(),
    )
    rp.validate()
    return rp


def normalize_transition_matrix(
    tm: pd.DataFrame,
    regime_order: list[str] | None = None,
) -> pd.DataFrame:
    if regime_order is None:
        regime_order = tm.index.astype(str).tolist()

    out = tm.copy().astype(float)

    for regime in regime_order:
        row_sum = float(out.loc[regime, regime_order].sum())
        if row_sum <= 0:
            raise ValueError(f"Transition row for regime '{regime}' has nonpositive total.")
        out.loc[regime, regime_order] = out.loc[regime, regime_order] / row_sum

    return out.loc[regime_order, regime_order].copy()


def sample_next_regime(
    current_regime: str,
    regime_process: RegimeProcess,
    rng: np.random.Generator | None = None,
) -> str:
    regime_process.validate()

    if current_regime not in regime_process.regimes:
        raise ValueError(
            f"Unknown current_regime '{current_regime}'. Valid regimes: {regime_process.regimes}"
        )

    if rng is None:
        rng = np.random.default_rng(123)

    probs = regime_process.row(current_regime).to_numpy()
    next_regime = rng.choice(regime_process.regimes, p=probs)
    return str(next_regime)


def simulate_regime_path(
    horizon: int,
    start_regime: str = "normal",
    regime_process: RegimeProcess | None = None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    if horizon <= 0:
        return []

    if regime_process is None:
        regime_process = build_default_regime_process()

    regime_process.validate()

    if start_regime not in regime_process.regimes:
        raise ValueError(
            f"Unknown start_regime '{start_regime}'. Valid regimes: {regime_process.regimes}"
        )

    if rng is None:
        rng = np.random.default_rng(123)

    path = [start_regime]
    current = start_regime

    for _ in range(1, horizon):
        current = sample_next_regime(current, regime_process, rng=rng)
        path.append(current)

    return path


def most_likely_next_regime(
    current_regime: str,
    regime_process: RegimeProcess | None = None,
) -> str:
    if regime_process is None:
        regime_process = build_default_regime_process()

    regime_process.validate()

    row = regime_process.row(current_regime)
    return str(row.idxmax())


def build_most_likely_regime_path(
    horizon: int,
    start_regime: str = "normal",
    regime_process: RegimeProcess | None = None,
) -> list[str]:
    if horizon <= 0:
        return []

    if regime_process is None:
        regime_process = build_default_regime_process()

    regime_process.validate()

    path = [start_regime]
    current = start_regime

    for _ in range(1, horizon):
        current = most_likely_next_regime(current, regime_process)
        path.append(current)

    return path


def regime_distribution_after_k_steps(
    start_regime: str,
    k: int,
    regime_process: RegimeProcess | None = None,
) -> pd.Series:
    if regime_process is None:
        regime_process = build_default_regime_process()

    regime_process.validate()

    if start_regime not in regime_process.regimes:
        raise ValueError(
            f"Unknown start_regime '{start_regime}'. Valid regimes: {regime_process.regimes}"
        )

    regimes = regime_process.regimes
    tm = regime_process.transition_matrix.loc[regimes, regimes].astype(float).to_numpy()

    p0 = np.zeros(len(regimes), dtype=float)
    p0[regimes.index(start_regime)] = 1.0

    if k == 0:
        return pd.Series(p0, index=regimes, name=f"dist_step_{k}")

    pk = p0 @ np.linalg.matrix_power(tm, k)
    return pd.Series(pk, index=regimes, name=f"dist_step_{k}")


def expected_regime_distributions_over_horizon(
    horizon: int,
    start_regime: str = "normal",
    regime_process: RegimeProcess | None = None,
) -> pd.DataFrame:
    if horizon <= 0:
        return pd.DataFrame()

    if regime_process is None:
        regime_process = build_default_regime_process()

    rows = []
    for k in range(horizon):
        dist = regime_distribution_after_k_steps(
            start_regime=start_regime,
            k=k,
            regime_process=regime_process,
        )
        row = {"step": k}
        for regime in regime_process.regimes:
            row[regime] = float(dist[regime])
        rows.append(row)

    return pd.DataFrame(rows)


def expected_regime_weights(
    current_regime: str,
    horizon: int,
    regime_process: RegimeProcess | None = None,
) -> dict[int, dict[str, float]]:
    """
    Returns a nested dict:
    {
        0: {"normal": ..., "surge": ..., "crisis": ...},
        1: {...},
        ...
    }
    """
    dist_df = expected_regime_distributions_over_horizon(
        horizon=horizon,
        start_regime=current_regime,
        regime_process=regime_process,
    )

    if dist_df.empty:
        return {}

    out: dict[int, dict[str, float]] = {}
    regime_cols = [c for c in dist_df.columns if c != "step"]

    for _, row in dist_df.iterrows():
        step = int(row["step"])
        out[step] = {regime: float(row[regime]) for regime in regime_cols}

    return out


def expected_multiplier_path(
    current_regime: str,
    horizon: int,
    multiplier_map: dict[str, float],
    regime_process: RegimeProcess | None = None,
) -> pd.DataFrame:
    """
    Computes expected multiplier at each future step under the regime Markov chain.

    Example multiplier_map:
    {
        "normal": 1.00,
        "surge": 1.20,
        "crisis": 1.45,
    }
    """
    if regime_process is None:
        regime_process = build_default_regime_process()

    regime_process.validate()

    missing = [g for g in regime_process.regimes if g not in multiplier_map]
    if missing:
        raise ValueError(
            f"Multiplier map missing entries for regimes: {missing}"
        )

    dist_df = expected_regime_distributions_over_horizon(
        horizon=horizon,
        start_regime=current_regime,
        regime_process=regime_process,
    )

    if dist_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in dist_df.iterrows():
        step = int(row["step"])
        expected_mult = 0.0
        component_weights = {}

        for regime in regime_process.regimes:
            prob = float(row[regime])
            mult = float(multiplier_map[regime])
            expected_mult += prob * mult
            component_weights[f"prob_{regime}"] = prob

        rows.append(
            {
                "step": step,
                "expected_multiplier": expected_mult,
                **component_weights,
            }
        )

    return pd.DataFrame(rows)


def regime_path_to_dataframe(path: Iterable[str]) -> pd.DataFrame:
    path_list = list(path)
    return pd.DataFrame(
        {
            "step": list(range(len(path_list))),
            "regime": path_list,
        }
    )


def print_regime_process_summary(
    regime_process: RegimeProcess | None = None,
    start_regime: str = "normal",
    horizon: int = 7,
) -> None:
    if regime_process is None:
        regime_process = build_default_regime_process()

    regime_process.validate()

    print("\nREGIME PROCESS SUMMARY")
    print("=" * 60)
    print("Regimes:", regime_process.regimes)
    print("\nTransition matrix:")
    print(regime_process.transition_matrix.loc[regime_process.regimes, regime_process.regimes])

    print(f"\nExpected regime distributions over horizon (start={start_regime}):")
    print(
        expected_regime_distributions_over_horizon(
            horizon=horizon,
            start_regime=start_regime,
            regime_process=regime_process,
        )
    )

    print(f"\nMost likely regime path (start={start_regime}):")
    print(
        build_most_likely_regime_path(
            horizon=horizon,
            start_regime=start_regime,
            regime_process=regime_process,
        )
    )