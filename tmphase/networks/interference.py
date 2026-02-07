"""Interference network — the core innovation of Phase Cancellation.

Takes two signals (excitatory and inhibitory positions in TruthSpace)
and produces a residual signal. The network learns context-dependent
interference: when to cancel fully, partially, or amplify.

Naive version: R(x) = E(x) + I(x)   (simple vector addition)
Learned version: R(x) = Interference(E(x), I(x))  (NEAT network)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..neat.genome import Genome, InnovationTracker, get_global_tracker
from ..neat.network import FeedForwardNetwork
from ..neat.population import Population


@dataclass
class InterferenceConfig:
    """Configuration for the Interference network."""

    position_dim: int = 16   # Each pathway produces this many dims
    output_dim: int = 16     # Residual position dimension
    population_size: int = 100
    output_activation: str = "tanh"

    @property
    def input_dim(self) -> int:
        return self.position_dim * 2  # Concatenation of E and I positions


class InterferenceMode:
    """Strategies for computing interference."""

    ADDITIVE = "additive"    # Simple vector addition (baseline)
    LEARNED = "learned"      # NEAT network learns interference


def additive_interference(pos_e: np.ndarray, pos_i: np.ndarray) -> np.ndarray:
    """Simple additive interference: R = E + I."""
    return pos_e + pos_i


class Interference:
    """Interference network that learns how signals should interact.

    Can operate in additive mode (simple addition, no learning) or
    learned mode (NEAT network determines interference dynamics).
    """

    def __init__(
        self,
        config: InterferenceConfig,
        mode: str = InterferenceMode.LEARNED,
        tracker: Optional[InnovationTracker] = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.tracker = tracker or get_global_tracker()

        if mode == InterferenceMode.LEARNED:
            self.population = Population(
                size=config.population_size,
                n_inputs=config.input_dim,
                n_outputs=config.output_dim,
                output_activation=config.output_activation,
                tracker=self.tracker,
            )
        else:
            self.population = None

    def interfere(self, pos_e: np.ndarray, pos_i: np.ndarray) -> np.ndarray:
        """Compute interference between excitatory and inhibitory signals.

        Returns the residual signal after interference.
        """
        if self.mode == InterferenceMode.ADDITIVE:
            return additive_interference(pos_e, pos_i)

        net = self.best_network
        if net is None:
            return additive_interference(pos_e, pos_i)

        combined = np.concatenate([pos_e, pos_i])
        return net.activate(combined)

    def interfere_with_genome(
        self, genome: Genome, pos_e: np.ndarray, pos_i: np.ndarray
    ) -> np.ndarray:
        """Compute interference using a specific genome."""
        if self.mode == InterferenceMode.ADDITIVE:
            return additive_interference(pos_e, pos_i)
        net = FeedForwardNetwork(genome)
        combined = np.concatenate([pos_e, pos_i])
        return net.activate(combined)

    @property
    def best_network(self) -> Optional[FeedForwardNetwork]:
        if self.population and self.population.best_genome:
            return FeedForwardNetwork(self.population.best_genome)
        return None

    def signal_strength(self, residual: np.ndarray) -> float:
        """Compute the strength of a residual signal (L2 norm)."""
        return float(np.linalg.norm(residual))

    def confidence(
        self, pos_e: np.ndarray, pos_i: np.ndarray, residual: np.ndarray
    ) -> float:
        """Compute confidence as |R| / max(|E|, |I|).

        confidence -> 1.0: constructive interference (agreement)
        confidence -> 0.0: destructive interference (cancellation)
        """
        e_norm = np.linalg.norm(pos_e)
        i_norm = np.linalg.norm(pos_i)
        r_norm = np.linalg.norm(residual)
        denom = max(e_norm, i_norm, 1e-8)
        return float(r_norm / denom)
