"""Boundary network — determines truth/falsehood from a TruthSpace position.

Takes a position (the residual after interference) and outputs a scalar
truth score in [0, 1]. The boundary is a learned decision surface in
TruthSpace, co-evolved with the embedders and interference network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..neat.genome import Genome, InnovationTracker, get_global_tracker
from ..neat.network import FeedForwardNetwork
from ..neat.population import Population


@dataclass
class BoundaryConfig:
    """Configuration for the Boundary network."""

    input_dim: int = 16      # TruthSpace position dimension
    population_size: int = 100
    output_activation: str = "sigmoid"  # Output in [0, 1]


class Boundary:
    """Boundary network: maps TruthSpace positions to truth scores."""

    def __init__(
        self,
        config: BoundaryConfig,
        tracker: Optional[InnovationTracker] = None,
    ) -> None:
        self.config = config
        self.tracker = tracker or get_global_tracker()
        self.population = Population(
            size=config.population_size,
            n_inputs=config.input_dim,
            n_outputs=1,
            output_activation=config.output_activation,
            tracker=self.tracker,
        )

    def evaluate(self, position: np.ndarray) -> float:
        """Evaluate truth score for a TruthSpace position using best network."""
        net = self.best_network
        if net is None:
            return 0.5  # Uncertain before evolution
        output = net.activate(position)
        return float(output[0])

    def evaluate_with_genome(self, genome: Genome, position: np.ndarray) -> float:
        """Evaluate using a specific genome."""
        net = FeedForwardNetwork(genome)
        output = net.activate(position)
        return float(output[0])

    @property
    def best_network(self) -> Optional[FeedForwardNetwork]:
        if self.population.best_genome:
            return FeedForwardNetwork(self.population.best_genome)
        return None
