"""Dual-pathway Embedder networks for Phase Cancellation.

Embedder_E (excitatory): Maps input symbols to positions that confirm truth.
Embedder_I (inhibitory): Maps input symbols to positions that contradict/negate.

Together they produce two signals that interfere — true statements survive,
false statements get cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..neat.genome import Genome, InnovationTracker, get_global_tracker
from ..neat.network import FeedForwardNetwork
from ..neat.population import Population


@dataclass
class EmbedderConfig:
    """Configuration for an Embedder network."""

    input_dim: int = 32   # Symbolic encoding dimension
    output_dim: int = 16  # TruthSpace position dimension
    population_size: int = 100
    output_activation: str = "tanh"


class Embedder:
    """Wrapper around a NEAT population that serves as an embedder.

    Can be either excitatory (E) or inhibitory (I) — the difference
    is in the fitness function, not the architecture.
    """

    def __init__(
        self,
        config: EmbedderConfig,
        role: str = "excitatory",
        tracker: Optional[InnovationTracker] = None,
    ) -> None:
        self.config = config
        self.role = role  # "excitatory" or "inhibitory"
        self.tracker = tracker or get_global_tracker()
        self.population = Population(
            size=config.population_size,
            n_inputs=config.input_dim,
            n_outputs=config.output_dim,
            output_activation=config.output_activation,
            tracker=self.tracker,
        )
        self._best_network: Optional[FeedForwardNetwork] = None

    @property
    def best_network(self) -> Optional[FeedForwardNetwork]:
        """The current best network in the population."""
        if self.population.best_genome is not None:
            return FeedForwardNetwork(self.population.best_genome)
        return self._best_network

    def embed(self, encoded_input: np.ndarray) -> np.ndarray:
        """Embed an encoded input using the best network. Returns TruthSpace position."""
        net = self.best_network
        if net is None:
            # Before any evolution, return zeros
            return np.zeros(self.config.output_dim, dtype=np.float64)
        return net.activate(encoded_input)

    def embed_with_genome(self, genome: Genome, encoded_input: np.ndarray) -> np.ndarray:
        """Embed using a specific genome (for fitness evaluation during evolution)."""
        net = FeedForwardNetwork(genome)
        return net.activate(encoded_input)
