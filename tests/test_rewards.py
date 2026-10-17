import pytest

from rl_rag.rewards import ndcg_reward


def test_perfect_ranking_scores_one() -> None:
    assert ndcg_reward([True, True], n_relevant=2, k=2) == pytest.approx(1.0)


def test_relevant_chunk_at_rank_two_scores_one_over_log2_three() -> None:
    # DCG = 1/log2(3); ideal DCG with one relevant chunk = 1/log2(2) = 1.
    assert ndcg_reward([False, True], n_relevant=1, k=2) == pytest.approx(1 / 1.5849625, rel=1e-6)


def test_no_matching_chunks_in_corpus_scores_zero() -> None:
    assert ndcg_reward([False, False], n_relevant=0, k=2) == 0.0


def test_k_larger_than_the_result_list_is_handled() -> None:
    assert ndcg_reward([True], n_relevant=1, k=8) == pytest.approx(1.0)


def test_reward_is_bounded_between_zero_and_one() -> None:
    value = ndcg_reward([True, False, True, False], n_relevant=5, k=4)
    assert 0.0 <= value <= 1.0
