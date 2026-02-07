"""Co-evolutionary training pipeline for the Phase Cancellation system.

All networks evolve together:
- Embedder_E and Embedder_I evolve adversarially
- Interference network learns context-dependent signal mixing
- Boundary network learns to read residual signals
- Each generation, the best of each network is used as the "opponent"
  for evaluating the other networks' fitness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..neat.genome import Genome, InnovationTracker, reset_global_tracker
from ..neat.network import FeedForwardNetwork
from ..utils.encoding import encode_statement
from ..neat.population import Population
from ..networks.embedder import Embedder, EmbedderConfig
from ..networks.interference import Interference, InterferenceConfig, InterferenceMode
from ..networks.boundary import Boundary, BoundaryConfig
from ..networks.dynamics import Dynamics, DynamicsConfig
from .fitness import (
    OracleDataset,
    fitness_embedder_excitatory,
    fitness_embedder_inhibitory,
    fitness_interference,
    fitness_boundary,
)


@dataclass
class SystemConfig:
    """Configuration for the full Phase Cancellation system."""

    input_dim: int = 32
    position_dim: int = 16
    population_size: int = 100
    use_learned_interference: bool = True
    elitism: int = 2
    survival_rate: float = 0.2
    crossover_rate: float = 0.75
    compatibility_threshold: float = 3.0


@dataclass
class EvolutionStats:
    """Statistics from one generation of evolution."""

    generation: int
    best_fitness_e: float
    best_fitness_i: float
    best_fitness_interference: float
    best_fitness_boundary: float
    accuracy: float  # Classification accuracy on oracle
    mean_true_signal: float
    mean_false_signal: float
    cancellation_ratio: float  # false_signal / true_signal (lower = better)


class PhaseCancellationSystem:
    """The full co-evolutionary Phase Cancellation system.

    Manages all 6 co-evolving populations (Layer 1 / TruthSpace):
    - Embedder_E (excitatory)
    - Embedder_I (inhibitory)
    - Interference
    - Boundary
    - Dynamics_fwd
    - Dynamics_ctr
    """

    def __init__(
        self,
        config: SystemConfig | None = None,
        oracle: OracleDataset | None = None,
    ) -> None:
        self.config = config or SystemConfig()
        self.oracle = oracle or OracleDataset.create_demo()

        # Pre-encode all oracle items
        for item in self.oracle.items:
            item.encode(self.config.input_dim)

        # Fresh innovation tracker for this system
        reset_global_tracker()
        from ..neat.genome import get_global_tracker
        self.tracker = get_global_tracker()

        cfg = self.config

        # Layer 1: TruthSpace networks
        self.embedder_e = Embedder(
            EmbedderConfig(
                input_dim=cfg.input_dim,
                output_dim=cfg.position_dim,
                population_size=cfg.population_size,
            ),
            role="excitatory",
            tracker=self.tracker,
        )
        self.embedder_i = Embedder(
            EmbedderConfig(
                input_dim=cfg.input_dim,
                output_dim=cfg.position_dim,
                population_size=cfg.population_size,
            ),
            role="inhibitory",
            tracker=self.tracker,
        )
        self.interference = Interference(
            InterferenceConfig(
                position_dim=cfg.position_dim,
                output_dim=cfg.position_dim,
                population_size=cfg.population_size,
            ),
            mode=(
                InterferenceMode.LEARNED
                if cfg.use_learned_interference
                else InterferenceMode.ADDITIVE
            ),
            tracker=self.tracker,
        )
        self.boundary = Boundary(
            BoundaryConfig(
                input_dim=cfg.position_dim,
                population_size=cfg.population_size,
            ),
            tracker=self.tracker,
        )
        self.dynamics_fwd = Dynamics(
            DynamicsConfig(
                position_dim=cfg.position_dim,
                output_dim=cfg.position_dim,
                population_size=cfg.population_size,
            ),
            role="forward",
            tracker=self.tracker,
        )
        self.dynamics_ctr = Dynamics(
            DynamicsConfig(
                position_dim=cfg.position_dim,
                output_dim=cfg.position_dim,
                population_size=cfg.population_size,
            ),
            role="contrary",
            tracker=self.tracker,
        )

        self.generation = 0
        self.stats_history: list[EvolutionStats] = []

    def _get_best_or_random(self, population: Population) -> Genome:
        """Get the best genome from the current generation, or a random one."""
        # Use the current generation's best (not all-time best) for co-evolution,
        # since fitness landscapes shift as opponents evolve.
        if population.genomes:
            return max(population.genomes, key=lambda g: g.fitness)
        if population.best_genome is not None:
            return population.best_genome
        return random.choice(population.genomes)

    def evolve_generation(self) -> EvolutionStats:
        """Run one generation of co-evolution across all networks."""
        cfg = self.config

        # Get current best opponents for fitness evaluation
        best_e = self._get_best_or_random(self.embedder_e.population)
        best_i = self._get_best_or_random(self.embedder_i.population)
        best_boundary = self._get_best_or_random(self.boundary.population)
        best_interference = (
            self._get_best_or_random(self.interference.population)
            if self.interference.population
            else None
        )

        # --- Evaluate Embedder_E ---
        def eval_e(genome: Genome) -> float:
            return fitness_embedder_excitatory(
                genome,
                embedder_i_genome=best_i,
                interference_genome=best_interference,
                boundary_genome=best_boundary,
                oracle=self.oracle,
                input_dim=cfg.input_dim,
                use_learned_interference=cfg.use_learned_interference,
            )

        self.embedder_e.population.evaluate(eval_e)
        self.embedder_e.population.evolve(
            elitism=cfg.elitism,
            survival_rate=cfg.survival_rate,
            crossover_rate=cfg.crossover_rate,
            compatibility_threshold=cfg.compatibility_threshold,
        )

        # Update best_e after evolution
        best_e = self._get_best_or_random(self.embedder_e.population)

        # --- Evaluate Embedder_I ---
        def eval_i(genome: Genome) -> float:
            return fitness_embedder_inhibitory(
                genome,
                embedder_e_genome=best_e,
                interference_genome=best_interference,
                boundary_genome=best_boundary,
                oracle=self.oracle,
                input_dim=cfg.input_dim,
                use_learned_interference=cfg.use_learned_interference,
            )

        self.embedder_i.population.evaluate(eval_i)
        self.embedder_i.population.evolve(
            elitism=cfg.elitism,
            survival_rate=cfg.survival_rate,
            crossover_rate=cfg.crossover_rate,
            compatibility_threshold=cfg.compatibility_threshold,
        )

        best_i = self._get_best_or_random(self.embedder_i.population)

        # --- Evaluate Interference (if learned) ---
        if self.interference.population is not None:
            def eval_interference(genome: Genome) -> float:
                return fitness_interference(
                    genome,
                    embedder_e_genome=best_e,
                    embedder_i_genome=best_i,
                    boundary_genome=best_boundary,
                    oracle=self.oracle,
                    input_dim=cfg.input_dim,
                )

            self.interference.population.evaluate(eval_interference)
            self.interference.population.evolve(
                elitism=cfg.elitism,
                survival_rate=cfg.survival_rate,
                crossover_rate=cfg.crossover_rate,
                compatibility_threshold=cfg.compatibility_threshold,
            )
            best_interference = self._get_best_or_random(self.interference.population)

        # --- Evaluate Boundary ---
        def eval_boundary(genome: Genome) -> float:
            return fitness_boundary(
                genome,
                embedder_e_genome=best_e,
                embedder_i_genome=best_i,
                interference_genome=best_interference,
                oracle=self.oracle,
                input_dim=cfg.input_dim,
                use_learned_interference=cfg.use_learned_interference,
            )

        self.boundary.population.evaluate(eval_boundary)
        self.boundary.population.evolve(
            elitism=cfg.elitism,
            survival_rate=cfg.survival_rate,
            crossover_rate=cfg.crossover_rate,
            compatibility_threshold=cfg.compatibility_threshold,
        )

        # --- Compute stats ---
        stats = self._compute_stats()
        self.stats_history.append(stats)
        self.generation += 1

        return stats

    def _compute_stats(self) -> EvolutionStats:
        """Compute statistics for the current generation."""
        best_e = self._get_best_or_random(self.embedder_e.population)
        best_i = self._get_best_or_random(self.embedder_i.population)
        best_boundary = self._get_best_or_random(self.boundary.population)
        best_interference = (
            self._get_best_or_random(self.interference.population)
            if self.interference.population
            else None
        )

        net_e = FeedForwardNetwork(best_e)
        net_i = FeedForwardNetwork(best_i)
        net_boundary = FeedForwardNetwork(best_boundary)
        net_interference = FeedForwardNetwork(best_interference) if best_interference else None

        correct = 0
        true_strengths = []
        false_strengths = []

        for item in self.oracle.items:
            encoded = item.encode(self.config.input_dim)
            pos_e = net_e.activate(encoded)
            pos_i = net_i.activate(encoded)

            if net_interference is not None and self.config.use_learned_interference:
                combined = np.concatenate([pos_e, pos_i])
                residual = net_interference.activate(combined)
            else:
                residual = pos_e + pos_i

            truth_score = net_boundary.activate(residual)[0]
            signal_strength = np.linalg.norm(residual)

            predicted_true = truth_score > 0.5
            if predicted_true == item.is_true:
                correct += 1

            if item.is_true:
                true_strengths.append(signal_strength)
            else:
                false_strengths.append(signal_strength)

        accuracy = correct / len(self.oracle) if self.oracle.items else 0.0
        mean_true = float(np.mean(true_strengths)) if true_strengths else 0.0
        mean_false = float(np.mean(false_strengths)) if false_strengths else 0.0
        cancellation_ratio = mean_false / (mean_true + 1e-8)

        return EvolutionStats(
            generation=self.generation,
            best_fitness_e=self.embedder_e.population.best_genome.fitness
            if self.embedder_e.population.best_genome
            else 0.0,
            best_fitness_i=self.embedder_i.population.best_genome.fitness
            if self.embedder_i.population.best_genome
            else 0.0,
            best_fitness_interference=(
                self.interference.population.best_genome.fitness
                if self.interference.population and self.interference.population.best_genome
                else 0.0
            ),
            best_fitness_boundary=self.boundary.population.best_genome.fitness
            if self.boundary.population.best_genome
            else 0.0,
            accuracy=accuracy,
            mean_true_signal=mean_true,
            mean_false_signal=mean_false,
            cancellation_ratio=cancellation_ratio,
        )

    def evaluate_statement(self, statement: str) -> dict:
        """Evaluate a single statement through the full Phase Cancellation pipeline.

        Returns a dict with truth_score, signal_strength, confidence, and prediction.
        """
        best_e = self._get_best_or_random(self.embedder_e.population)
        best_i = self._get_best_or_random(self.embedder_i.population)
        best_boundary = self._get_best_or_random(self.boundary.population)
        best_interference = (
            self._get_best_or_random(self.interference.population)
            if self.interference.population
            else None
        )

        net_e = FeedForwardNetwork(best_e)
        net_i = FeedForwardNetwork(best_i)
        net_boundary = FeedForwardNetwork(best_boundary)
        net_interference = FeedForwardNetwork(best_interference) if best_interference else None

        encoded = encode_statement(statement, self.config.input_dim)
        pos_e = net_e.activate(encoded)
        pos_i = net_i.activate(encoded)

        if net_interference is not None and self.config.use_learned_interference:
            combined = np.concatenate([pos_e, pos_i])
            residual = net_interference.activate(combined)
        else:
            residual = pos_e + pos_i

        truth_score = float(net_boundary.activate(residual)[0])
        signal_strength = float(np.linalg.norm(residual))
        e_norm = float(np.linalg.norm(pos_e))
        i_norm = float(np.linalg.norm(pos_i))
        confidence = signal_strength / max(e_norm, i_norm, 1e-8)

        return {
            "statement": statement,
            "truth_score": truth_score,
            "signal_strength": signal_strength,
            "confidence": confidence,
            "predicted_true": truth_score > 0.5,
            "pos_excitatory_norm": e_norm,
            "pos_inhibitory_norm": i_norm,
        }

    def train(
        self,
        generations: int = 50,
        verbose: bool = True,
    ) -> list[EvolutionStats]:
        """Train the system for a number of generations."""
        for gen in range(generations):
            stats = self.evolve_generation()
            if verbose:
                print(
                    f"Gen {stats.generation:4d} | "
                    f"Acc: {stats.accuracy:.2%} | "
                    f"True signal: {stats.mean_true_signal:.3f} | "
                    f"False signal: {stats.mean_false_signal:.3f} | "
                    f"Cancel ratio: {stats.cancellation_ratio:.3f} | "
                    f"Fit E: {stats.best_fitness_e:.3f} "
                    f"I: {stats.best_fitness_i:.3f} "
                    f"Int: {stats.best_fitness_interference:.3f} "
                    f"Bnd: {stats.best_fitness_boundary:.3f}"
                )

        return self.stats_history
