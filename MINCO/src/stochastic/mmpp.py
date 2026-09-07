"""Markov-modulated Poisson demand model for MINCO.

The hidden state captures an operational demand regime and the emission model
captures integer arrivals in each period.  This module intentionally avoids a
third-party HMM dependency so the mathematics is visible and testable.

For hidden regime Z_t and arrivals Y_t:

    P(Z_t=j | Z_{t-1}=i) = A[i,j]
    Y_t | Z_t=k ~ Poisson(lambda[k])

Baum-Welch (EM) is used for unsupervised parameter estimation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import gammaln, logsumexp


_EPS = 1e-12


@dataclass(frozen=True)
class PoissonHMMFit:
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    rates: np.ndarray
    log_likelihood_history: tuple[float, ...]

    @property
    def n_states(self) -> int:
        return int(self.rates.shape[0])


def _as_counts(values: Iterable[int | float]) -> np.ndarray:
    y = np.asarray(list(values), dtype=float)
    if y.ndim != 1 or y.size < 3:
        raise ValueError("At least three one-dimensional observations are required")
    if np.any(~np.isfinite(y)) or np.any(y < 0) or np.any(np.abs(y - np.rint(y)) > 1e-9):
        raise ValueError("Poisson observations must be finite nonnegative integers")
    return y.astype(int)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.maximum(np.asarray(vector, dtype=float), _EPS)
    return vector / vector.sum()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(np.asarray(matrix, dtype=float), _EPS)
    return matrix / matrix.sum(axis=1, keepdims=True)


def poisson_log_emission(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Return log p(y_t | state=k) with shape (T, K)."""
    y = np.asarray(counts, dtype=float)[:, None]
    lam = np.maximum(np.asarray(rates, dtype=float)[None, :], _EPS)
    return y * np.log(lam) - lam - gammaln(y + 1.0)


def _forward_backward(
    counts: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    rates: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scaled log-domain forward/backward posterior computation."""
    log_b = poisson_log_emission(counts, rates)
    log_pi = np.log(np.maximum(initial, _EPS))
    log_a = np.log(np.maximum(transition, _EPS))
    t_count, n_states = log_b.shape

    log_alpha = np.empty((t_count, n_states), dtype=float)
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, t_count):
        log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)
    log_likelihood = float(logsumexp(log_alpha[-1]))

    log_beta = np.zeros((t_count, n_states), dtype=float)
    for t in range(t_count - 2, -1, -1):
        log_beta[t] = logsumexp(log_a + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.empty((t_count - 1, n_states, n_states), dtype=float)
    for t in range(t_count - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_a
            + log_b[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - log_likelihood
        )
        xi_t = np.exp(log_xi)
        xi[t] = xi_t / xi_t.sum()
    return log_likelihood, gamma, xi


def fit_poisson_hmm(
    observations: Iterable[int | float],
    *,
    n_states: int = 4,
    max_iter: int = 200,
    tolerance: float = 1e-6,
    random_state: int = 20260817,
) -> PoissonHMMFit:
    """Estimate a Poisson HMM using Baum-Welch.

    State labels are sorted by fitted Poisson rate before returning, which makes
    regime interpretation stable: state 0 is lowest demand and the last state is
    highest demand.
    """
    y = _as_counts(observations)
    if not 2 <= n_states <= min(8, y.size):
        raise ValueError("n_states must be between 2 and min(8, number of observations)")
    if max_iter < 2:
        raise ValueError("max_iter must be at least 2")

    rng = np.random.default_rng(random_state)
    positive_scale = max(float(y.mean()), 1.0)
    quantiles = np.quantile(y, np.linspace(0.1, 0.9, n_states))
    rates = np.maximum(quantiles + rng.normal(0.0, 0.03 * positive_scale, n_states), 0.1)
    transition = np.full((n_states, n_states), 0.15 / max(n_states - 1, 1), dtype=float)
    np.fill_diagonal(transition, 0.85)
    transition = _normalize_rows(transition)
    initial = np.full(n_states, 1.0 / n_states, dtype=float)

    history: list[float] = []
    for _ in range(max_iter):
        ll, gamma, xi = _forward_backward(y, initial, transition, rates)
        history.append(ll)

        initial = _normalize_vector(gamma[0])
        transition = _normalize_rows(xi.sum(axis=0))
        weights = gamma.sum(axis=0)
        rates = np.maximum((gamma * y[:, None]).sum(axis=0) / np.maximum(weights, _EPS), 0.05)

        if len(history) >= 2 and abs(history[-1] - history[-2]) <= tolerance * (1.0 + abs(history[-2])):
            break

    # Final posterior after the last M step so reported likelihood matches parameters.
    ll, _, _ = _forward_backward(y, initial, transition, rates)
    if not history or abs(ll - history[-1]) > 1e-12:
        history.append(ll)

    order = np.argsort(rates)
    return PoissonHMMFit(
        initial_probabilities=_normalize_vector(initial[order]),
        transition_matrix=_normalize_rows(transition[np.ix_(order, order)]),
        rates=rates[order],
        log_likelihood_history=tuple(float(v) for v in history),
    )


def posterior_regime_probabilities(observations: Iterable[int | float], fit: PoissonHMMFit) -> np.ndarray:
    y = _as_counts(observations)
    _, gamma, _ = _forward_backward(y, fit.initial_probabilities, fit.transition_matrix, fit.rates)
    return gamma


def simulate_mmpp(
    *,
    transition_matrix: np.ndarray,
    rates: np.ndarray,
    n_periods: int,
    initial_probabilities: np.ndarray | None = None,
    seed: int = 20260817,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate hidden regimes and Poisson arrivals with deterministic seeding."""
    a = _normalize_rows(np.asarray(transition_matrix, dtype=float))
    lam = np.asarray(rates, dtype=float)
    if a.shape[0] != a.shape[1] or a.shape[0] != lam.size:
        raise ValueError("Transition matrix and rates have incompatible dimensions")
    if np.any(lam <= 0) or n_periods <= 0:
        raise ValueError("Rates and n_periods must be positive")
    pi = _normalize_vector(initial_probabilities if initial_probabilities is not None else np.ones(lam.size))

    rng = np.random.default_rng(seed)
    states = np.empty(n_periods, dtype=int)
    counts = np.empty(n_periods, dtype=int)
    states[0] = int(rng.choice(lam.size, p=pi))
    counts[0] = int(rng.poisson(lam[states[0]]))
    for t in range(1, n_periods):
        states[t] = int(rng.choice(lam.size, p=a[states[t - 1]]))
        counts[t] = int(rng.poisson(lam[states[t]]))
    return states, counts


def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    a = _normalize_rows(np.asarray(transition_matrix, dtype=float))
    if a.shape[0] != a.shape[1]:
        raise ValueError("Transition matrix must be square")
    eigenvalues, eigenvectors = np.linalg.eig(a.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vector = np.real(eigenvectors[:, index])
    if vector.sum() < 0:
        vector *= -1
    return _normalize_vector(vector)
