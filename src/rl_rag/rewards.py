"""Reward definition: retrieval quality of one arm on one query.

The reward is binary-relevance nDCG@k, computed by the RAG evaluation library
from the same source-level labels the library uses for its reports. Nothing
here calls an LLM, so computing rewards costs no money and is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragplatform.evals.metrics import ndcg_at_k

if TYPE_CHECKING:
    from collections.abc import Sequence


def ndcg_reward(relevance: Sequence[bool], n_relevant: int, k: int) -> float:
    """nDCG@k in [0, 1]; 0.0 when the labels match no chunk in the corpus.

    ``relevance`` is the relevance flag of each retrieved chunk in rank order,
    and ``n_relevant`` is how many corpus chunks match the labels.
    """
    return ndcg_at_k(relevance, n_relevant, k)
