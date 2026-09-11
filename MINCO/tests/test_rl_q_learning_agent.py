from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.rl.q_learning_agent import QLearningAgent

ACTIONS = ["local_only", "myopic_milp", "optimized_network"]


def test_new_agent_has_zeroed_q_table_for_unseen_states() -> None:
    agent = QLearningAgent(ACTIONS, random_seed=1)
    q_vals = agent.q_table["never_seen_state"]
    assert q_vals.shape == (3,)
    assert np.all(q_vals == 0.0)


def test_select_action_always_explores_when_epsilon_is_one() -> None:
    agent = QLearningAgent(ACTIONS, epsilon=1.0, random_seed=1)
    agent.q_table["s"] = np.array([10.0, 0.0, 0.0])  # action 0 is clearly best
    seen = {agent.select_action("s", explore=True) for _ in range(200)}
    # With epsilon=1.0 every call is a random draw -- all three actions should show up.
    assert seen == {0, 1, 2}


def test_select_action_is_greedy_when_epsilon_is_zero() -> None:
    agent = QLearningAgent(ACTIONS, epsilon=0.0, random_seed=1)
    agent.q_table["s"] = np.array([1.0, 5.0, 2.0])
    for _ in range(20):
        assert agent.select_action("s", explore=True) == 1
    assert agent.select_action("s", explore=False) == 1


def test_select_action_breaks_ties_among_equal_max_q_values() -> None:
    agent = QLearningAgent(ACTIONS, epsilon=0.0, random_seed=1)
    agent.q_table["s"] = np.array([4.0, 4.0, 0.0])
    seen = {agent.select_action("s", explore=True) for _ in range(200)}
    # Both tied-for-best actions (0 and 1) should be selectable; the strictly worse one (2) never.
    assert seen == {0, 1}


def test_greedy_action_picks_argmax_deterministically() -> None:
    agent = QLearningAgent(ACTIONS, random_seed=1)
    agent.q_table["s"] = np.array([1.0, 9.0, 3.0])
    assert agent.greedy_action("s") == 1


def test_update_applies_bellman_equation_for_non_terminal_transition() -> None:
    agent = QLearningAgent(ACTIONS, alpha=0.5, gamma=0.9, random_seed=1)
    agent.q_table["s0"][0] = 2.0
    agent.q_table["s1"] = np.array([1.0, 4.0, 0.0])  # max is 4.0

    agent.update(state_key="s0", action_idx=0, reward=10.0, next_state_key="s1", done=False)

    target = 10.0 + 0.9 * 4.0
    expected = 2.0 + 0.5 * (target - 2.0)
    assert agent.q_table["s0"][0] == pytest.approx(expected)


def test_update_ignores_next_state_value_when_done() -> None:
    agent = QLearningAgent(ACTIONS, alpha=0.5, gamma=0.9, random_seed=1)
    agent.q_table["s0"][0] = 2.0
    agent.q_table["s1"] = np.array([100.0, 100.0, 100.0])  # would dominate if not ignored

    agent.update(state_key="s0", action_idx=0, reward=10.0, next_state_key="s1", done=True)

    expected = 2.0 + 0.5 * (10.0 - 2.0)  # target is just the reward, no bootstrapped next value
    assert agent.q_table["s0"][0] == pytest.approx(expected)


def test_update_only_touches_the_acted_action_index() -> None:
    agent = QLearningAgent(ACTIONS, alpha=1.0, gamma=0.0, random_seed=1)
    agent.update(state_key="s0", action_idx=1, reward=5.0, next_state_key="s1", done=True)
    assert agent.q_table["s0"][1] == pytest.approx(5.0)
    assert agent.q_table["s0"][0] == 0.0
    assert agent.q_table["s0"][2] == 0.0


def test_decay_epsilon_multiplies_and_floors_at_minimum() -> None:
    agent = QLearningAgent(ACTIONS, epsilon=1.0, epsilon_min=0.4, epsilon_decay=0.5, random_seed=1)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    agent.decay_epsilon()
    # 0.5 * 0.5 = 0.25, which is below epsilon_min=0.4 -- must clamp, not keep decaying.
    assert agent.epsilon == pytest.approx(0.4)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.4)


def test_to_dataframe_reports_best_action_and_value_per_state() -> None:
    agent = QLearningAgent(ACTIONS, random_seed=1)
    agent.q_table["s0"] = np.array([1.0, 9.0, 3.0])
    agent.q_table["s1"] = np.array([0.0, 0.0, 0.0])

    df = agent.to_dataframe()
    row = df.set_index("state_key").loc["s0"]
    assert row["best_action"] == "myopic_milp"
    assert row["best_q_value"] == pytest.approx(9.0)
    assert row["Q::optimized_network"] == pytest.approx(3.0)


def test_to_dataframe_on_empty_q_table_returns_expected_columns() -> None:
    agent = QLearningAgent(ACTIONS, random_seed=1)
    df = agent.to_dataframe()
    assert list(df.columns) == ["state_key", "best_action", "best_q_value"]
    assert len(df) == 0


def test_save_and_load_pickle_round_trips_q_table_and_hyperparameters(tmp_path: Path) -> None:
    agent = QLearningAgent(ACTIONS, alpha=0.3, gamma=0.8, epsilon=0.6, epsilon_min=0.1,
                            epsilon_decay=0.99, random_seed=7)
    agent.q_table["s0"] = np.array([1.0, 2.0, 3.0])
    agent.q_table["s1"] = np.array([4.0, 5.0, 6.0])

    path = tmp_path / "agent.pkl"
    agent.save_pickle(path)
    restored = QLearningAgent.load_pickle(path)

    assert restored.action_names == agent.action_names
    assert restored.alpha == agent.alpha
    assert restored.gamma == agent.gamma
    assert restored.epsilon == agent.epsilon
    assert restored.epsilon_min == agent.epsilon_min
    assert restored.epsilon_decay == agent.epsilon_decay
    assert restored.random_seed == agent.random_seed
    np.testing.assert_array_equal(restored.q_table["s0"], agent.q_table["s0"])
    np.testing.assert_array_equal(restored.q_table["s1"], agent.q_table["s1"])
    # Unseen states must still zero-initialize after a round trip, not raise KeyError.
    np.testing.assert_array_equal(restored.q_table["never_seen"], np.zeros(3))
