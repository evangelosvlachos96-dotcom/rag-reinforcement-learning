import numpy as np

from rl_rag.simulate import greedy_arm, paired_bootstrap_gap, run_policy, split_queries
from rl_rag.bandits import EpsilonGreedy


def test_split_is_a_partition_of_the_queries() -> None:
    train, test = split_queries(10, seed=0)
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(10))
    assert train.size == 5 and test.size == 5


def test_bootstrap_gap_is_positive_when_one_arm_dominates() -> None:
    matrix = np.array([[0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])
    result = paired_bootstrap_gap(matrix, ["good", "bad"], resamples=200)
    assert result["best_arm"] == "good"
    assert result["mean_gap"] == 0.8
    assert result["ci95_low"] > 0


def test_bootstrap_gap_straddles_zero_for_identical_arms() -> None:
    matrix = np.array([[0.5, 0.2, 0.8], [0.5, 0.2, 0.8]])
    result = paired_bootstrap_gap(matrix, ["a", "b"], resamples=200)
    assert result["mean_gap"] == 0.0
    assert result["ci95_low"] <= 0.0 <= result["ci95_high"]


def test_epsilon_greedy_commits_to_the_best_arm_on_a_clear_matrix() -> None:
    matrix = np.array([[0.1, 0.1], [0.9, 0.9], [0.3, 0.3]])
    regret, policy = run_policy(
        lambda k: EpsilonGreedy(k, epsilon=0.05), matrix, np.arange(2), rounds=300, seed=1
    )
    assert greedy_arm(policy) == 1
    assert regret[-1] >= 0.0
