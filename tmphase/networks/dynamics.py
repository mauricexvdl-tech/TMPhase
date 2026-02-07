"""Dynamics networks — forward inference and contrary inference.

Dynamics_fwd: Given two TruthSpace positions (premises), infer a new position
              (conclusion) via forward reasoning.
Dynamics_ctr: Given the same premises, infer what FALSE conclusion might be
              drawn — the adversarial inference path.

The interference of fwd and ctr filters out fallacious conclusions intrinsically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..neat.genome import Genome, InnovationTracker, get_global_tracker
from ..neat.network import FeedForwardNetwork
from ..neat.population import Population


@dataclass
class DynamicsConfig:
    """Configuration for Dynamics networks."""

    position_dim: int = 16   # Each premise is a TruthSpace position
    output_dim: int = 16     # Conclusion is also a TruthSpace position
    population_size: int = 100
    output_activation: str = "tanh"

    @property
    def input_dim(self) -> int:
        return self.position_dim * 2  # Two premises concatenated


class Dynamics:
    """Dynamics network for inference in TruthSpace.

    Can be either 'forward' (standard inference) or 'contrary' (adversarial).
    """

    def __init__(
        self,
        config: DynamicsConfig,
        role: str = "forward",
        tracker: Optional[InnovationTracker] = None,
    ) -> None:
        self.config = config
        self.role = role  # "forward" or "contrary"
        self.tracker = tracker or get_global_tracker()
        self.population = Population(
            size=config.population_size,
            n_inputs=config.input_dim,
            n_outputs=config.output_dim,
            output_activation=config.output_activation,
            tracker=self.tracker,
        )

    def infer(self, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """Infer a new position from two premise positions."""
        net = self.best_network
        if net is None:
            return np.zeros(self.config.output_dim, dtype=np.float64)
        combined = np.concatenate([pos_a, pos_b])
        return net.activate(combined)

    def infer_with_genome(
        self, genome: Genome, pos_a: np.ndarray, pos_b: np.ndarray
    ) -> np.ndarray:
        """Infer using a specific genome."""
        net = FeedForwardNetwork(genome)
        combined = np.concatenate([pos_a, pos_b])
        return net.activate(combined)

    @property
    def best_network(self) -> Optional[FeedForwardNetwork]:
        if self.population.best_genome:
            return FeedForwardNetwork(self.population.best_genome)
        return None
