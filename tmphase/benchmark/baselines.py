"""Baseline classifiers for comparing against Phase Cancellation.

1. RandomBaseline     — coin flip (theoretical lower bound)
2. SinglePathNEAT     — one embedder + boundary (no cancellation)
3. EnsembleNEAT       — two independent embedders averaged + boundary
4. PhaseCancellation  — the full dual-path interference system
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..neat.genome import Genome, InnovationTracker
from ..neat.network import FeedForwardNetwork
from ..neat.population import Population
from ..evolution.fitness import OracleDataset
from ..utils.encoding import encode_statement


@dataclass
class BaselineResult:
    """Result from a baseline evaluation."""

    name: str
    train_accuracy: float
    test_accuracy: float
    generations: int


# ── Baseline 1: Random ──────────────────────────────────────────────

class RandomBaseline:
    """Always predicts randomly. Theoretical 50% accuracy."""

    def evaluate(
        self,
        train: OracleDataset,
        test: OracleDataset,
        generations: int = 50,
        seed: int = 42,
    ) -> BaselineResult:
        import random
        rng = random.Random(seed)

        train_correct = sum(1 for _ in train.items if rng.random() > 0.5)
        test_correct = sum(1 for _ in test.items if rng.random() > 0.5)

        return BaselineResult(
            name="Random",
            train_accuracy=train_correct / len(train),
            test_accuracy=test_correct / len(test),
            generations=0,
        )


# ── Baseline 2: Single-Path NEAT ────────────────────────────────────

class SinglePathNEAT:
    """One embedder + one boundary. No cancellation, no dual path.

    This is the control: same compute budget (roughly) without interference.
    """

    def __init__(
        self,
        input_dim: int = 32,
        position_dim: int = 16,
        population_size: int = 100,
    ):
        self.input_dim = input_dim
        self.position_dim = position_dim
        self.population_size = population_size

    def evaluate(
        self,
        train: OracleDataset,
        test: OracleDataset,
        generations: int = 50,
        seed: int = 42,
    ) -> BaselineResult:
        import random
        random.seed(seed)
        np.random.seed(seed)

        tracker = InnovationTracker()
        embedder_pop = Population(
            size=self.population_size,
            n_inputs=self.input_dim,
            n_outputs=self.position_dim,
            output_activation="tanh",
            tracker=tracker,
        )
        boundary_pop = Population(
            size=self.population_size,
            n_inputs=self.position_dim,
            n_outputs=1,
            output_activation="sigmoid",
            tracker=tracker,
        )

        # Pre-encode
        for item in train.items:
            item.encode(self.input_dim)
        for item in test.items:
            item.encode(self.input_dim)

        best_emb: Optional[Genome] = None
        best_bnd: Optional[Genome] = None

        for gen in range(generations):
            # Get current best opponents
            cur_bnd = best_bnd if best_bnd else random.choice(boundary_pop.genomes)

            # Evaluate embedder
            def eval_emb(genome: Genome, _bnd=cur_bnd) -> float:
                net_e = FeedForwardNetwork(genome)
                net_b = FeedForwardNetwork(_bnd)
                score = 0.0
                for item in train.items:
                    pos = net_e.activate(item.encode(self.input_dim))
                    ts = net_b.activate(pos)[0]
                    score += ts if item.is_true else (1.0 - ts)
                return max(0.001, score / len(train))

            embedder_pop.evaluate(eval_emb)
            embedder_pop.evolve()
            best_emb = max(embedder_pop.genomes, key=lambda g: g.fitness)

            # Evaluate boundary
            def eval_bnd(genome: Genome, _emb=best_emb) -> float:
                net_e = FeedForwardNetwork(_emb)
                net_b = FeedForwardNetwork(genome)
                score = 0.0
                for item in train.items:
                    pos = net_e.activate(item.encode(self.input_dim))
                    ts = net_b.activate(pos)[0]
                    score += ts if item.is_true else (1.0 - ts)
                return max(0.001, score / len(train))

            boundary_pop.evaluate(eval_bnd)
            boundary_pop.evolve()
            best_bnd = max(boundary_pop.genomes, key=lambda g: g.fitness)

        # Evaluate on train and test
        net_e = FeedForwardNetwork(best_emb)
        net_b = FeedForwardNetwork(best_bnd)

        def accuracy(dataset: OracleDataset) -> float:
            correct = 0
            for item in dataset.items:
                pos = net_e.activate(item.encode(self.input_dim))
                ts = net_b.activate(pos)[0]
                if (ts > 0.5) == item.is_true:
                    correct += 1
            return correct / len(dataset)

        return BaselineResult(
            name="SinglePath",
            train_accuracy=accuracy(train),
            test_accuracy=accuracy(test),
            generations=generations,
        )


# ── Baseline 3: Ensemble NEAT ───────────────────────────────────────

class EnsembleNEAT:
    """Two independent embedders, outputs averaged, then boundary.

    Same parameter count as Phase Cancellation but without adversarial
    training or interference — just a simple ensemble.
    """

    def __init__(
        self,
        input_dim: int = 32,
        position_dim: int = 16,
        population_size: int = 100,
    ):
        self.input_dim = input_dim
        self.position_dim = position_dim
        self.population_size = population_size

    def evaluate(
        self,
        train: OracleDataset,
        test: OracleDataset,
        generations: int = 50,
        seed: int = 42,
    ) -> BaselineResult:
        import random
        random.seed(seed)
        np.random.seed(seed)

        tracker = InnovationTracker()
        emb_a_pop = Population(
            size=self.population_size,
            n_inputs=self.input_dim,
            n_outputs=self.position_dim,
            output_activation="tanh",
            tracker=tracker,
        )
        emb_b_pop = Population(
            size=self.population_size,
            n_inputs=self.input_dim,
            n_outputs=self.position_dim,
            output_activation="tanh",
            tracker=tracker,
        )
        boundary_pop = Population(
            size=self.population_size,
            n_inputs=self.position_dim,
            n_outputs=1,
            output_activation="sigmoid",
            tracker=tracker,
        )

        for item in train.items:
            item.encode(self.input_dim)
        for item in test.items:
            item.encode(self.input_dim)

        best_a: Optional[Genome] = None
        best_b: Optional[Genome] = None
        best_bnd: Optional[Genome] = None

        for gen in range(generations):
            cur_b = best_b if best_b else random.choice(emb_b_pop.genomes)
            cur_bnd = best_bnd if best_bnd else random.choice(boundary_pop.genomes)

            # Evaluate embedder A
            def eval_a(genome: Genome, _b=cur_b, _bnd=cur_bnd) -> float:
                net_a = FeedForwardNetwork(genome)
                net_b = FeedForwardNetwork(_b)
                net_bnd = FeedForwardNetwork(_bnd)
                score = 0.0
                for item in train.items:
                    enc = item.encode(self.input_dim)
                    pos = (net_a.activate(enc) + net_b.activate(enc)) / 2.0
                    ts = net_bnd.activate(pos)[0]
                    score += ts if item.is_true else (1.0 - ts)
                return max(0.001, score / len(train))

            emb_a_pop.evaluate(eval_a)
            emb_a_pop.evolve()
            best_a = max(emb_a_pop.genomes, key=lambda g: g.fitness)

            # Evaluate embedder B
            def eval_b(genome: Genome, _a=best_a, _bnd=cur_bnd) -> float:
                net_a = FeedForwardNetwork(_a)
                net_b = FeedForwardNetwork(genome)
                net_bnd = FeedForwardNetwork(_bnd)
                score = 0.0
                for item in train.items:
                    enc = item.encode(self.input_dim)
                    pos = (net_a.activate(enc) + net_b.activate(enc)) / 2.0
                    ts = net_bnd.activate(pos)[0]
                    score += ts if item.is_true else (1.0 - ts)
                return max(0.001, score / len(train))

            emb_b_pop.evaluate(eval_b)
            emb_b_pop.evolve()
            best_b = max(emb_b_pop.genomes, key=lambda g: g.fitness)

            # Evaluate boundary
            def eval_bnd(genome: Genome, _a=best_a, _b=best_b) -> float:
                net_a = FeedForwardNetwork(_a)
                net_b = FeedForwardNetwork(_b)
                net_bnd = FeedForwardNetwork(genome)
                score = 0.0
                for item in train.items:
                    enc = item.encode(self.input_dim)
                    pos = (net_a.activate(enc) + net_b.activate(enc)) / 2.0
                    ts = net_bnd.activate(pos)[0]
                    score += ts if item.is_true else (1.0 - ts)
                return max(0.001, score / len(train))

            boundary_pop.evaluate(eval_bnd)
            boundary_pop.evolve()
            best_bnd = max(boundary_pop.genomes, key=lambda g: g.fitness)

        # Final evaluation
        net_a = FeedForwardNetwork(best_a)
        net_b = FeedForwardNetwork(best_b)
        net_bnd = FeedForwardNetwork(best_bnd)

        def accuracy(dataset: OracleDataset) -> float:
            correct = 0
            for item in dataset.items:
                enc = item.encode(self.input_dim)
                pos = (net_a.activate(enc) + net_b.activate(enc)) / 2.0
                ts = net_bnd.activate(pos)[0]
                if (ts > 0.5) == item.is_true:
                    correct += 1
            return correct / len(dataset)

        return BaselineResult(
            name="Ensemble",
            train_accuracy=accuracy(train),
            test_accuracy=accuracy(test),
            generations=generations,
        )


# ── Wrapper for Phase Cancellation ──────────────────────────────────

class PhaseCancellationBaseline:
    """Wraps the full Phase Cancellation system as a baseline entry."""

    def __init__(
        self,
        input_dim: int = 32,
        position_dim: int = 16,
        population_size: int = 100,
        use_learned_interference: bool = True,
    ):
        self.input_dim = input_dim
        self.position_dim = position_dim
        self.population_size = population_size
        self.use_learned_interference = use_learned_interference

    def evaluate(
        self,
        train: OracleDataset,
        test: OracleDataset,
        generations: int = 50,
        seed: int = 42,
    ) -> BaselineResult:
        import random
        random.seed(seed)
        np.random.seed(seed)

        from ..evolution.coevolution import PhaseCancellationSystem, SystemConfig

        config = SystemConfig(
            input_dim=self.input_dim,
            position_dim=self.position_dim,
            population_size=self.population_size,
            use_learned_interference=self.use_learned_interference,
        )
        system = PhaseCancellationSystem(config=config, oracle=train)
        system.train(generations=generations, verbose=False)

        # Train accuracy from last stats
        train_acc = system.stats_history[-1].accuracy if system.stats_history else 0.0

        # Test accuracy
        test_correct = 0
        for item in test.items:
            result = system.evaluate_statement(item.statement)
            if result["predicted_true"] == item.is_true:
                test_correct += 1
        test_acc = test_correct / len(test) if test.items else 0.0

        name = "PhaseCancel" if self.use_learned_interference else "PhaseCancel(add)"
        return BaselineResult(
            name=name,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            generations=generations,
        )
