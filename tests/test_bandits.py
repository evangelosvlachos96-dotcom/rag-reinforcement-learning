import numpy as np
import pytest

from rl_rag.bandits import UCB1, EpsilonGreedy, RandomPolicy


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_every_arm_is_tried_once_before_exploiting() -> None:
    policy = UCB1(n_arms=3)
    r = rng()
    seen = []
    for t in range(1, 4):
        arm = policy.select(t, r)
        seen.append(arm)
        policy.update(arm, reward=0.0)
    assert sorted(seen) == [0, 1, 2]


def test_epsilon_greedy_exploits_the_best_arm_when_epsilon_is_zero() -> None:
    policy = EpsilonGreedy(n_arms=3, epsilon=0.0)
    r = rng()
    for arm, reward in [(0, 0.2), (1, 0.9), (2, 0.5)]:
        policy.update(arm, reward)
    assert all(policy.select(t, r) == 1 for t in range(4, 10))


def test_running_mean_is_updated_correctly() -> None:
    policy = EpsilonGreedy(n_arms=1, epsilon=0.0)
    for reward in [1.0, 0.0, 0.5]:
        policy.update(0, reward)
    assert policy._stats.means[0] == pytest.approx(0.5)
    assert policy._stats.counts[0] == 3


def test_ucb_prefers_a_less_tried_arm_with_equal_means() -> None:
    policy = UCB1(n_arms=2)
    r = rng()
    for _ in range(50):
        policy.update(0, 0.5)
    policy.update(1, 0.5)
    assert policy.select(t=52, rng=r) == 1


def test_ucb_converges_to_the_clearly_best_arm() -> None:
    means = np.array([0.2, 0.8, 0.4])
    r = rng(1)
    policy = UCB1(n_arms=3)
    picks = []
    for t in range(1, 401):
        arm = policy.select(t, r)
        policy.update(arm, float(means[arm]))
        picks.append(arm)
    assert picks[-100:].count(1) >= 90


def test_epsilon_greedy_explores_roughly_epsilon_of_the_time() -> None:
    policy = EpsilonGreedy(n_arms=4, epsilon=0.2)
    r = rng(2)
    for arm in range(4):
        policy.update(arm, 1.0 if arm == 0 else 0.0)
    picks = [policy.select(t, r) for t in range(10_000)]
    non_best = sum(1 for a in picks if a != 0) / len(picks)
    # Exploring picks a random arm, so 3/4 of exploration draws are non-best.
    assert non_best == pytest.approx(0.2 * 0.75, abs=0.02)


def test_random_policy_covers_all_arms() -> None:
    policy = RandomPolicy(n_arms=5)
    r = rng(3)
    picks = {policy.select(t, r) for t in range(200)}
    assert picks == {0, 1, 2, 3, 4}


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        EpsilonGreedy(n_arms=2, epsilon=1.5)
    with pytest.raises(ValueError):
        UCB1(n_arms=0)
