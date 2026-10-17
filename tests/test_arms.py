import pytest

from rl_rag.arms import Arm, build_arms


def test_default_grid_has_nine_arms_in_stable_order() -> None:
    arms = build_arms()
    assert len(arms) == 9
    assert arms[0] == Arm(chunk_size=200, k=4)
    assert arms[-1] == Arm(chunk_size=800, k=16)
    assert build_arms() == arms


def test_arm_ids_are_unique_and_readable() -> None:
    ids = [arm.arm_id for arm in build_arms()]
    assert len(set(ids)) == len(ids)
    assert "chunk400_k8" in ids


def test_custom_grid() -> None:
    arms = build_arms(chunk_sizes=[100], k_values=[1, 2])
    assert [a.arm_id for a in arms] == ["chunk100_k1", "chunk100_k2"]


def test_empty_grid_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_arms(chunk_sizes=[], k_values=[4])


def test_arm_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError):
        Arm(chunk_size=0, k=4)
