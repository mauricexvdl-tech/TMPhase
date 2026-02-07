"""Entry point for running TMPhase training from the command line.

Usage:
    python -m tmphase [--generations N] [--population N] [--no-learned-interference]
"""

from __future__ import annotations

import argparse
import sys

from .evolution.coevolution import PhaseCancellationSystem, SystemConfig
from .evolution.fitness import OracleDataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TMPhase — Phase Cancellation Architecture training"
    )
    parser.add_argument(
        "--generations", type=int, default=50, help="Number of generations to train"
    )
    parser.add_argument(
        "--population", type=int, default=100, help="Population size per network"
    )
    parser.add_argument(
        "--input-dim", type=int, default=32, help="Input encoding dimension"
    )
    parser.add_argument(
        "--position-dim", type=int, default=16, help="TruthSpace position dimension"
    )
    parser.add_argument(
        "--no-learned-interference",
        action="store_true",
        help="Use additive interference instead of learned",
    )
    args = parser.parse_args(argv)

    config = SystemConfig(
        input_dim=args.input_dim,
        position_dim=args.position_dim,
        population_size=args.population,
        use_learned_interference=not args.no_learned_interference,
    )

    oracle = OracleDataset.create_demo()

    print("=" * 72)
    print("TMPhase — Phase Cancellation Architecture")
    print("Optimierung durch Destruktion")
    print("=" * 72)
    print(f"Config: {config}")
    print(f"Oracle: {len(oracle)} items ({len(oracle.true_items)} true, {len(oracle.false_items)} false)")
    print(f"Interference mode: {'learned' if config.use_learned_interference else 'additive'}")
    print("=" * 72)
    print()

    system = PhaseCancellationSystem(config=config, oracle=oracle)
    system.train(generations=args.generations, verbose=True)

    # Show final evaluation on some test statements
    print()
    print("=" * 72)
    print("Final evaluation:")
    print("=" * 72)

    test_statements = [
        "water is wet",
        "the sun is cold",
        "fire burns",
        "ice is hot",
        "birds can fly",
        "rocks can fly",
        "gravity pulls things down",
        "gravity pushes things up",
    ]

    for stmt in test_statements:
        result = system.evaluate_statement(stmt)
        label = "TRUE " if result["predicted_true"] else "FALSE"
        print(
            f"  [{label}] {result['truth_score']:.3f} "
            f"(signal: {result['signal_strength']:.3f}, "
            f"conf: {result['confidence']:.3f}) "
            f"| {stmt}"
        )


if __name__ == "__main__":
    main()
