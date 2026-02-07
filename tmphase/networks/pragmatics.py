"""Pragmatics layer — Reachability, Utility, and Planner networks.

These operate above the TruthSpace layer, using truth-evaluated positions
to make decisions about action.

Reachability: Given a position, which other positions can be reached?
Utility:      Given a position + context, how useful is it?
Planner:      Given current position + goal, what is the next step?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..neat.genome import Genome, InnovationTracker, get_global_tracker
from ..neat.network import FeedForwardNetwork
from ..neat.population import Population


@dataclass
class ReachabilityConfig:
    """Configuration for the Reachability network."""

    input_dim: int = 16       # Current TruthSpace position
    n_reachable: int = 5      # How many reachable positions to output
    position_dim: int = 16
    population_size: int = 100
    output_activation: str = "tanh"

    @property
    def output_dim(self) -> int:
        return self.n_reachable * self.position_dim  # 5 × 16 = 80


class Reachability:
    """Maps a TruthSpace position to a set of reachable positions."""

    def __init__(
        self,
        config: ReachabilityConfig,
        tracker: Optional[InnovationTracker] = None,
    ) -> None:
        self.config = config
        self.tracker = tracker or get_global_tracker()
        self.population = Population(
            size=config.population_size,
            n_inputs=config.input_dim,
            n_outputs=config.output_dim,
            output_activation=config.output_activation,
            tracker=self.tracker,
        )

    def get_reachable(self, position: np.ndarray) -> list[np.ndarray]:
        """Get reachable positions from the given TruthSpace position."""
        net = self.best_network
        if net is None:
            return [np.zeros(self.config.position_dim) for _ in range(self.config.n_reachable)]
        raw = net.activate(position)
        # Split into individual positions
        positions = []
        for i in range(self.config.n_reachable):
            start = i * self.config.position_dim
            end = start + self.config.position_dim
            positions.append(raw[start:end])
        return positions

    @property
    def best_network(self) -> Optional[FeedForwardNetwork]:
        if self.population.best_genome:
            return FeedForwardNetwork(self.population.best_genome)
        return None


@dataclass
class UtilityConfig:
    """Configuration for the Utility network."""

    input_dim: int = 32  # Position (16) + context (16)
    population_size: int = 100
    output_activation: str = "sigmoid"  # Output in [0, 1]


class Utility:
    """Evaluates the utility/usefulness of a TruthSpace position in context."""

    def __init__(
        self,
        config: UtilityConfig,
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

    def evaluate(self, position: np.ndarray, context: np.ndarray) -> float:
        """Evaluate utility of a position given a context."""
        net = self.best_network
        if net is None:
            return 0.5
        combined = np.concatenate([position, context])
        return float(net.activate(combined)[0])

    @property
    def best_network(self) -> Optional[FeedForwardNetwork]:
        if self.population.best_genome:
            return FeedForwardNetwork(self.population.best_genome)
        return None


@dataclass
class PlannerConfig:
    """Configuration for the Planner network."""

    input_dim: int = 32      # Current position (16) + goal position (16)
    output_dim: int = 16     # Next step position
    population_size: int = 100
    output_activation: str = "tanh"


class Planner:
    """Plans the next step from a current position toward a goal position."""

    def __init__(
        self,
        config: PlannerConfig,
        tracker: Optional[InnovationTracker] = None,
    ) -> None:
        self.config = config
        self.tracker = tracker or get_global_tracker()
        self.population = Population(
            size=config.population_size,
            n_inputs=config.input_dim,
            n_outputs=config.output_dim,
            output_activation=config.output_activation,
            tracker=self.tracker,
        )

    def next_step(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Plan the next step from current toward goal."""
        net = self.best_network
        if net is None:
            return np.zeros(self.config.output_dim, dtype=np.float64)
        combined = np.concatenate([current, goal])
        return net.activate(combined)

    @property
    def best_network(self) -> Optional[FeedForwardNetwork]:
        if self.population.best_genome:
            return FeedForwardNetwork(self.population.best_genome)
        return None
