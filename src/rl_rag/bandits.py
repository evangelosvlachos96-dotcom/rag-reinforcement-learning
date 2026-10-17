"""Stateless multi-armed bandit policies, implemented from scratch.

A policy keeps a running mean reward per arm. ``select`` picks an arm given
the round number ``t`` (starting at 1) and a random generator; ``update``
feeds back the observed reward. Every arm is tried once before any policy
uses its statistics, so no arm is ever ranked on zero observations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


class Policy(Protocol):
    """What the simulator needs from any bandit algorithm."""

    name: str

    def select(self, t: int, rng: Generator) -> int: ...

    def update(self, arm: int, reward: float) -> None: ...


class _Stats:
    """Pull counts and running mean reward per arm."""

    def __init__(self, n_arms: int) -> None:
        if n_arms < 1:
            raise ValueError("a bandit needs at least one arm")
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.means = np.zeros(n_arms, dtype=np.float64)

    def record(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]

    def first_untried(self) -> int | None:
        untried = np.flatnonzero(self.counts == 0)
        return int(untried[0]) if untried.size else None


class RandomPolicy:
    """Uniform random choice. The baseline any learning policy must beat."""

    name = "random"

    def __init__(self, n_arms: int) -> None:
        self._stats = _Stats(n_arms)

    def select(self, t: int, rng: Generator) -> int:
        return int(rng.integers(len(self._stats.counts)))

    def update(self, arm: int, reward: float) -> None:
        self._stats.record(arm, reward)


class EpsilonGreedy:
    """With probability ``epsilon`` explore a random arm, otherwise exploit the best mean."""

    def __init__(self, n_arms: int, epsilon: float = 0.1) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.name = f"epsilon_greedy(eps={epsilon:g})"
        self._stats = _Stats(n_arms)

    def select(self, t: int, rng: Generator) -> int:
        untried = self._stats.first_untried()
        if untried is not None:
            return untried
        if rng.random() < self.epsilon:
            return int(rng.integers(len(self._stats.counts)))
        return _argmax_random_tie(self._stats.means, rng)

    def update(self, arm: int, reward: float) -> None:
        self._stats.record(arm, reward)


class UCB1:
    """Upper confidence bound: mean plus sqrt(2 ln t / n), optimism under uncertainty."""

    name = "ucb1"

    def __init__(self, n_arms: int) -> None:
        self._stats = _Stats(n_arms)

    def select(self, t: int, rng: Generator) -> int:
        untried = self._stats.first_untried()
        if untried is not None:
            return untried
        bonus = np.sqrt(2.0 * math.log(max(t, 2)) / self._stats.counts)
        return _argmax_random_tie(self._stats.means + bonus, rng)

    def update(self, arm: int, reward: float) -> None:
        self._stats.record(arm, reward)


def _argmax_random_tie(values: np.ndarray, rng: Generator) -> int:
    """Index of the maximum, choosing uniformly among exact ties."""
    best = np.flatnonzero(values == values.max())
    return int(best[rng.integers(best.size)]) if best.size > 1 else int(best[0])
