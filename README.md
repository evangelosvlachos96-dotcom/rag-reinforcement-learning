# rag-reinforcement-learning

Multi-armed bandits that learn which RAG retrieval configuration to use, rewarded by the retrieval metrics of the [agentic-rag-eval-platform](https://github.com/evangelosvlachos96-dotcom/agentic-rag-eval-platform) library.

A RAG system has knobs that change answer quality: how the corpus is chunked and how many chunks reach the generator. Choosing them by hand means guessing or running a full grid. This project treats the choice as a reinforcement learning problem. Each configuration is an *arm*, retrieval quality on a query is the *reward*, and a bandit policy learns which arm to favour while still exploring the others.

## Why this is reinforcement learning

It is the simplest RL setting: a stateless bandit. An agent picks an action, receives a reward, and updates its estimate of each action's value. It faces the exploration-exploitation trade-off that underlies RL in general. Adding query types as context, which would make it a contextual bandit, is listed under next steps.

## What is implemented

| Component | Module | Notes |
| --- | --- | --- |
| Arms | `src/rl_rag/arms.py` | 9 arms: chunk size (200, 400, 800 tokens) × context size k (4, 8, 16) |
| Policies | `src/rl_rag/bandits.py` | Random baseline, epsilon-greedy, UCB1, all implemented from scratch |
| Reward | `src/rl_rag/rewards.py` | Binary-relevance nDCG@k, from the RAG library's metric code |
| Reward computation | `src/rl_rag/build_rewards.py` | Ingests the corpus per chunk size, builds BM25, scores every (arm, query) pair |
| Simulation | `src/rl_rag/simulate.py` | Regret over many seeded runs, held-out check, paired bootstrap CI, plots |

Tests cover the arms, the policies, the reward function and the simulation helpers (22 tests).

## Results

The reward is nDCG@k on the 20 answerable questions of the RAG repo's `v1` eval set, using BM25 retrieval over the pinned PEP corpus (56 documents). Each cell is the mean over those 20 queries.

| chunk size | k = 4 | k = 8 | k = 16 |
| --- | --- | --- | --- |
| 200 tokens | 0.269 | 0.301 | 0.328 |
| 400 tokens | 0.359 | 0.380 | 0.397 |
| 800 tokens | 0.328 | 0.378 | **0.419** |

The `chunk400_k8` value (0.380) matches the `ndcg@8` the RAG library reports for its own BM25 run, so the reward pipeline agrees with the library's reporting.

Simulation: 500 rounds, 200 seeded runs per policy, queries drawn from a 10-query training half.

| policy | mean cumulative regret | final arm chosen most often |
| --- | --- | --- |
| random | 33.7 ± 1.0 | `chunk800_k16` (42% of runs) |
| epsilon-greedy (ε = 0.1) | 14.3 ± 9.2 | `chunk800_k16` (53% of runs) |
| UCB1 | 29.8 ± 1.3 | `chunk800_k16` (54% of runs) |

Regret is measured against the best fixed arm. Epsilon-greedy learns fastest here. UCB1 explores longer because its bonus assumes rewards spread over [0, 1], while the gaps between these arms are only 0.02 to 0.05. See `results/regret.png` and `results/arm_rewards.png`.

**What the results do not show:**

- **The winning arm is not statistically established.** `chunk800_k16` leads the runner-up (`chunk400_k16`) by 0.022 nDCG, but a paired bootstrap over the 20 queries gives a 95% interval of **-0.018 to +0.065**, which includes zero. With 20 queries, the ranking is a hypothesis to test on a bigger eval set, not a finding.
- **The held-out check is small.** Policies trained on 10 queries picked `chunk800_k16` and scored 0.305 nDCG on the other 10 (the best arm on that half scored 0.323, and a random arm averaged 0.253). Ten queries per side is too few to say much.
- **The reward measures retrieval, not answers.** nDCG@k rewards larger k by construction. It says nothing about answer faithfulness, generation cost or latency, which the RAG library measures with LLM judges that this project does not run.
- **Vector and hybrid retrieval are not included.** Those configurations need an embedding model downloaded from Hugging Face, which was not reachable from the environment where this was built. The arms are BM25-only for that reason.

## Reproduce

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone the RAG library next to this repo and fetch the pinned PEP corpus
git clone https://github.com/evangelosvlachos96-dotcom/agentic-rag-eval-platform ../agentic-rag-eval-platform
cd ../agentic-rag-eval-platform && uv run rag download-corpus && cd -

# 2. Install this project (pulls the library at a pinned commit)
uv sync --extra dev

# 3. Compute the rewards (writes results/rewards.json; uses no LLM calls)
uv run rl-rag-build-rewards --rag-root ../agentic-rag-eval-platform

# 4. Simulate the policies (writes results/simulation.json and the plots)
uv run rl-rag-simulate

# 5. Tests and lint
uv run pytest
uv run ruff check src tests
```

Step 3 writes the processed corpus to `data/processed/` (git-ignored). `results/rewards.json` records the library commit, eval-set version and dataset version of every chunk size, so the numbers can be traced back to their inputs.

## Limitations and next steps

- Only 20 answerable queries. The next step is a larger held-out eval set. The RAG library's candidate-and-review pipeline can produce one, but its candidate generation makes paid model calls, so it needs approval first.
- Add the vector and hybrid arms on a machine that can download the embedding model, then check whether the ranking changes.
- Use query type as context (a contextual bandit), so the policy can learn different arms for different kinds of question.
- Replace the nDCG reward with the library's answer-faithfulness and correctness judges, which costs money per call and needs approval first.
- UCB1 needs a reward scale suited to these gaps, or more rounds, before it competes with epsilon-greedy here.

## Project layout

```
src/rl_rag/          arms, bandits, rewards, reward builder, simulator
tests/               unit tests for each module
results/             rewards.json, simulation.json, regret.png, arm_rewards.png
docs/                GitHub Pages site
```

## Dependencies

- [agentic-rag-eval-platform](https://github.com/evangelosvlachos96-dotcom/agentic-rag-eval-platform), pinned to commit `7837dbf`, for corpus ingestion, BM25 retrieval, eval-set loading and the nDCG metric.
- numpy and matplotlib for the bandits and plots.

## License

MIT. See the project metadata in `pyproject.toml`.
