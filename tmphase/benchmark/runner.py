"""Benchmark runner — compares all approaches with statistical rigor.

Runs multiple seeds, reports mean +/- std for each approach on each dataset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .datasets import (
    create_easy_dataset,
    create_medium_dataset,
    create_hard_dataset,
    create_split_dataset,
)
from .baselines import (
    RandomBaseline,
    SinglePathNEAT,
    EnsembleNEAT,
    PhaseCancellationBaseline,
    BaselineResult,
)
from .analysis import analyze_cancellation, format_analysis
from ..evolution.coevolution import PhaseCancellationSystem, SystemConfig


@dataclass
class BenchmarkConfig:
    input_dim: int = 32
    position_dim: int = 16
    population_size: int = 50
    generations: int = 30
    n_seeds: int = 5
    train_ratio: float = 0.7


def run_benchmark(config: BenchmarkConfig | None = None) -> None:
    """Run the full benchmark suite."""
    config = config or BenchmarkConfig()

    datasets = {
        "EASY": create_easy_dataset(),
        "MEDIUM": create_medium_dataset(),
        "HARD": create_hard_dataset(),
    }

    approaches = {
        "Random": RandomBaseline(),
        "SinglePath": SinglePathNEAT(
            input_dim=config.input_dim,
            position_dim=config.position_dim,
            population_size=config.population_size,
        ),
        "Ensemble": EnsembleNEAT(
            input_dim=config.input_dim,
            position_dim=config.position_dim,
            population_size=config.population_size,
        ),
        "PhaseCancel": PhaseCancellationBaseline(
            input_dim=config.input_dim,
            position_dim=config.position_dim,
            population_size=config.population_size,
            use_learned_interference=True,
        ),
        "PhaseCancel(add)": PhaseCancellationBaseline(
            input_dim=config.input_dim,
            position_dim=config.position_dim,
            population_size=config.population_size,
            use_learned_interference=False,
        ),
    }

    print("=" * 72)
    print("TMPhase Benchmark Suite")
    print(f"Config: pop={config.population_size}, gen={config.generations}, "
          f"seeds={config.n_seeds}, train_ratio={config.train_ratio}")
    print("=" * 72)

    for ds_name, dataset in datasets.items():
        print(f"\n{'─' * 72}")
        print(f"Dataset: {ds_name} ({len(dataset)} items: "
              f"{len(dataset.true_items)} true, {len(dataset.false_items)} false)")
        print(f"{'─' * 72}")

        # Collect results per approach
        results: dict[str, list[BaselineResult]] = {name: [] for name in approaches}

        for seed in range(config.n_seeds):
            train, test = create_split_dataset(dataset, config.train_ratio, seed=seed)
            print(f"\n  Seed {seed}: train={len(train)}, test={len(test)}")

            for app_name, approach in approaches.items():
                t0 = time.time()
                result = approach.evaluate(
                    train=train,
                    test=test,
                    generations=config.generations,
                    seed=seed,
                )
                elapsed = time.time() - t0
                results[app_name].append(result)
                print(f"    {app_name:20s} | "
                      f"train: {result.train_accuracy:.1%}  "
                      f"test: {result.test_accuracy:.1%}  "
                      f"({elapsed:.1f}s)")

        # Summary table
        print(f"\n  {'─' * 60}")
        print(f"  Summary ({ds_name}):")
        print(f"  {'Approach':20s} | {'Train':>14s} | {'Test':>14s}")
        print(f"  {'─' * 20}-+-{'─' * 14}-+-{'─' * 14}")

        for app_name, res_list in results.items():
            train_accs = [r.train_accuracy for r in res_list]
            test_accs = [r.test_accuracy for r in res_list]
            print(f"  {app_name:20s} | "
                  f"{np.mean(train_accs):.1%} +/- {np.std(train_accs):.1%} | "
                  f"{np.mean(test_accs):.1%} +/- {np.std(test_accs):.1%}")

    # Cancellation analysis on medium dataset
    print(f"\n{'=' * 72}")
    print("Cancellation Analysis (MEDIUM dataset, full data, last seed)")
    print(f"{'=' * 72}")

    medium = create_medium_dataset()
    for item in medium.items:
        item.encode(config.input_dim)

    sys_config = SystemConfig(
        input_dim=config.input_dim,
        position_dim=config.position_dim,
        population_size=config.population_size,
        use_learned_interference=True,
    )
    system = PhaseCancellationSystem(config=sys_config, oracle=medium)
    system.train(generations=config.generations, verbose=False)

    metrics = analyze_cancellation(system, medium)
    print(format_analysis(metrics))


if __name__ == "__main__":
    run_benchmark()
