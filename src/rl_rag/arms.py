"""The action space: one arm per (chunk size, context size) pair.

An arm is a retrieval configuration the bandit can pick. Chunk size changes
the corpus index (it requires re-chunking), so arms that share a chunk size
share one index. Context size ``k`` is how many retrieved chunks the generator
would receive, which is also the cutoff used to score the arm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CHUNK_SIZES: tuple[int, ...] = (200, 400, 800)
DEFAULT_K_VALUES: tuple[int, ...] = (4, 8, 16)


class Arm(BaseModel):
    """One retrieval configuration: chunk token budget and number of chunks used."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: int = Field(gt=0, description="Token budget per chunk.")
    k: int = Field(gt=0, description="Chunks passed to the generator (the nDCG cutoff).")

    @property
    def arm_id(self) -> str:
        return f"chunk{self.chunk_size}_k{self.k}"


def build_arms(
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> list[Arm]:
    """Every combination of chunk size and k, in a stable order."""
    if not chunk_sizes or not k_values:
        raise ValueError("need at least one chunk size and one k value")
    return [Arm(chunk_size=c, k=k) for c in chunk_sizes for k in k_values]
