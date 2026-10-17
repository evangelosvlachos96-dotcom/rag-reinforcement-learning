"""Simulate the bandit policies on the precomputed rewards and write results.

Environment: each round a query is drawn uniformly from the training queries,
and the chosen arm earns its precomputed nDCG@k on that query. The expected
reward of an arm is its mean over training queries, so the best fixed arm is
the oracle and regret is measured against it.

Held-out check: the policies only see the training half of the queries. The
final greedy arm of each policy is then scored on the test half, against the
best arm on that half. This guards against picking an arm just because it
happened to fit the queries it was trained on.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from rl_rag.bandits import UCB1, EpsilonGreedy, Policy, RandomPolicy

if TYPE_CHECKING:
    from collections.abc import Callable


def load_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """Arm ids and an (arms x queries) reward matrix from ``rewards.json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    arm_ids = [arm["arm_id"] for arm in payload["arms"]]
    matrix = np.array([payload["rewards"][a] for a in arm_ids], dtype=np.float64)
    return arm_ids, matrix


def split_queries(n: int, seed: int, train_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    cut = int(round(n * train_fraction))
    return np.sort(order[:cut]), np.sort(order[cut:])


def run_policy(
    make_policy: Callable[[int], Policy],
    matrix: np.ndarray,
    train_queries: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[np.ndarray, Policy]:
    """Expected regret per round for one run, plus the trained policy."""
    rng = np.random.default_rng(seed)
    n_arms = matrix.shape[0]
    policy = make_policy(n_arms)
    means = matrix[:, train_queries].mean(axis=1)
    best = means.max()
    regret = np.empty(rounds)
    for t in range(1, rounds + 1):
        arm = policy.select(t, rng)
        query = train_queries[rng.integers(train_queries.size)]
        policy.update(arm, float(matrix[arm, query]))
        regret[t - 1] = best - means[arm]
    return regret, policy


def greedy_arm(policy: Policy) -> int:
    """The arm a policy would commit to now: highest observed mean."""
    stats = policy._stats  # noqa: SLF001  (simulation reads the learned means)
    return int(np.argmax(stats.means))


def paired_bootstrap_gap(
    matrix: np.ndarray, arm_ids: list[str], resamples: int = 10_000, seed: int = 0
) -> dict[str, object]:
    """95% CI for the mean reward gap between the two best arms, resampling queries.

    Both arms are scored on the same queries, so the paired difference is the
    right quantity: it cancels out how hard each sampled query set happens to be.
    """
    means = matrix.mean(axis=1)
    first, second = np.argsort(means)[::-1][:2]
    diff = matrix[first] - matrix[second]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    boot = diff[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "best_arm": arm_ids[int(first)],
        "runner_up_arm": arm_ids[int(second)],
        "mean_gap": float(diff.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": resamples,
        "n_queries": int(diff.size),
    }


def simulate(
    matrix: np.ndarray,
    arm_ids: list[str],
    rounds: int,
    runs: int,
    split_seed: int,
) -> dict[str, object]:
    n_arms, n_queries = matrix.shape
    train, test = split_queries(n_queries, seed=split_seed)
    factories: dict[str, Callable[[int], Policy]] = {
        "random": RandomPolicy,
        "epsilon_greedy_0.1": lambda k: EpsilonGreedy(k, epsilon=0.1),
        "ucb1": UCB1,
    }
    train_means = matrix[:, train].mean(axis=1)
    test_means = matrix[:, test].mean(axis=1)
    summary: dict[str, object] = {
        "rounds": rounds,
        "runs": runs,
        "split_seed": split_seed,
        "train_queries": int(train.size),
        "test_queries": int(test.size),
        "best_fixed_arm_train": arm_ids[int(np.argmax(train_means))],
        "best_fixed_arm_test": arm_ids[int(np.argmax(test_means))],
        "policies": {},
    }
    curves: dict[str, np.ndarray] = {}
    for name, factory in factories.items():
        all_regret = np.empty((runs, rounds))
        chosen_test_scores = []
        final_arms: list[int] = []
        for run in range(runs):
            regret, policy = run_policy(factory, matrix, train, rounds, seed=run)
            all_regret[run] = regret
            arm = greedy_arm(policy)
            final_arms.append(arm)
            chosen_test_scores.append(float(test_means[arm]))
        curves[name] = np.cumsum(all_regret, axis=1).mean(axis=0)
        values, counts = np.unique(final_arms, return_counts=True)
        summary["policies"][name] = {  # type: ignore[index]
            "mean_cumulative_regret": float(curves[name][-1]),
            "std_cumulative_regret": float(np.cumsum(all_regret, axis=1)[:, -1].std()),
            "most_common_final_arm": arm_ids[int(values[np.argmax(counts)])],
            "final_arm_share": float(counts.max() / runs),
            "held_out_mean_nDCG_of_final_arm": float(np.mean(chosen_test_scores)),
        }
    summary["best_vs_runner_up"] = paired_bootstrap_gap(matrix, arm_ids)
    summary["held_out_mean_nDCG_random_arm"] = float(test_means.mean())
    summary["held_out_mean_nDCG_best_arm"] = float(test_means.max())
    summary["train_mean_nDCG_per_arm"] = dict(zip(arm_ids, train_means.tolist(), strict=True))
    summary["test_mean_nDCG_per_arm"] = dict(zip(arm_ids, test_means.tolist(), strict=True))
    return summary, curves


def write_plots(
    curves: dict[str, np.ndarray], train_means: dict[str, float], out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=name)
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative expected regret (nDCG)")
    ax.set_title("Bandit regret vs. the best fixed arm (training queries)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "regret.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(train_means), list(train_means.values()))
    ax.set_ylabel("mean nDCG (training queries)")
    ax.set_title("Reward of each arm")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "arm_rewards.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rewards", type=Path, default=Path("results/rewards.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args(argv)

    arm_ids, matrix = load_matrix(args.rewards)
    summary, curves = simulate(matrix, arm_ids, args.rounds, args.runs, args.split_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_plots(curves, summary["train_mean_nDCG_per_arm"], args.out_dir)  # type: ignore[arg-type]
    summary["generated_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    summary["rewards_file"] = str(args.rewards)
    (args.out_dir / "simulation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["policies"], indent=2))  # noqa: T201
    print(f"best fixed arm (train): {summary['best_fixed_arm_train']}")  # noqa: T201
    print(f"best fixed arm (test):  {summary['best_fixed_arm_test']}")  # noqa: T201


if __name__ == "__main__":
    main()
