"""Compute the reward of every (arm, query) pair and write it to a JSON file.

For each chunk size the script ingests the RAG repo's PEP corpus, builds the
BM25 index, retrieves the top candidates for every answerable eval item, and
scores each context size k with nDCG@k. Only BM25 is used: the vector and
hybrid configurations need an embedding model download, which this script
does not perform.

Run the RAG repo's ``rag download-corpus`` once before this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ragplatform.evals.eval_set import load_eval_set
from ragplatform.evals.relevance import count_relevant_chunks, relevance_flags
from ragplatform.ingestion.chunking import ChunkingConfig
from ragplatform.ingestion.dataset import read_chunks
from ragplatform.ingestion.pipeline import ingest_directory
from ragplatform.retrieval.config import RetrievalConfig
from ragplatform.retrieval.index import build_index, load_retriever

from rl_rag.arms import DEFAULT_CHUNK_SIZES, DEFAULT_K_VALUES, build_arms
from rl_rag.rewards import ndcg_reward

SCHEMA_VERSION = 1


def _rag_commit(rag_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(rag_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def compute_rewards(
    rag_root: Path,
    work_dir: Path,
    eval_set: str,
    chunk_sizes: list[int],
    k_values: list[int],
    candidate_k: int,
) -> dict[str, object]:
    raw_dir = rag_root / "data" / "raw" / "peps"
    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} missing; run `rag download-corpus` in the RAG repo")
    items, eval_version = load_eval_set(rag_root / "eval_sets", eval_set)
    queries = [item for item in items if item.answerable]
    arms = build_arms(chunk_sizes, k_values)

    rewards: dict[str, list[float]] = {}
    dataset_versions: dict[str, str] = {}
    zero_label_queries: dict[str, int] = {}
    for chunk_size in chunk_sizes:
        ingested = ingest_directory(raw_dir, work_dir, ChunkingConfig(max_tokens=chunk_size))
        dataset_versions[str(chunk_size)] = ingested.dataset_version
        build_index(ingested.output_dir, None)
        chunks = read_chunks(ingested.output_dir)
        retriever = load_retriever(
            RetrievalConfig(mode="bm25", candidate_k=candidate_k, final_k=candidate_k),
            ingested.output_dir,
        )
        per_k: dict[int, list[float]] = {k: [] for k in k_values}
        no_labels = 0
        for item in queries:
            retrieved = retriever.retrieve(item.question)
            flags = relevance_flags(retrieved, item.relevant_sources)
            n_relevant = count_relevant_chunks(chunks, item.relevant_sources)
            no_labels += n_relevant == 0
            for k in k_values:
                per_k[k].append(ndcg_reward(flags, n_relevant, k))
        zero_label_queries[str(chunk_size)] = no_labels
        for k in k_values:
            rewards[f"chunk{chunk_size}_k{k}"] = per_k[k]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": {
            "rag_repo_commit": _rag_commit(rag_root),
            "rag_library_pin": "7837dbfbcb40e6a42549bec85a10ee29a78e275c",
            "eval_set": eval_set,
            "eval_set_version": eval_version,
            "retriever": "bm25",
            "metric": "binary-relevance nDCG@k",
            "candidate_k": candidate_k,
            "dataset_versions_by_chunk_size": dataset_versions,
            "queries_with_no_matching_chunks_by_chunk_size": zero_label_queries,
        },
        "arms": [
            {
                "arm_id": arm.arm_id,
                "chunk_size": arm.chunk_size,
                "k": arm.k,
                "dataset_version": dataset_versions[str(arm.chunk_size)],
            }
            for arm in arms
        ],
        "queries": [item.id for item in queries],
        "rewards": rewards,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rag-root", type=Path, default=Path("../agentic-rag-eval-platform"))
    parser.add_argument("--eval-set", default="v1")
    parser.add_argument("--work-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("results/rewards.json"))
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=list(DEFAULT_CHUNK_SIZES))
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--candidate-k", type=int, default=50)
    args = parser.parse_args(argv)

    payload = compute_rewards(
        rag_root=args.rag_root.resolve(),
        work_dir=args.work_dir,
        eval_set=args.eval_set,
        chunk_sizes=args.chunk_sizes,
        k_values=args.k_values,
        candidate_k=args.candidate_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} with {len(payload['arms'])} arms")  # noqa: T201


if __name__ == "__main__":
    main()
