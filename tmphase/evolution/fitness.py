"""Co-evolutionary fitness functions for Phase Cancellation networks.

The key insight: Embedder_E and Embedder_I co-evolve AGAINST each other.
- Embedder_E is rewarded when TRUE statements survive cancellation.
- Embedder_I is rewarded when FALSE statements are cancelled.
- The Interference network learns context-dependent interference.
- The Boundary network learns to read the residual signals.

This creates a natural arms race (similar to GANs but via evolution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..neat.genome import Genome
from ..neat.network import FeedForwardNetwork
from ..utils.encoding import encode_statement


@dataclass
class OracleItem:
    """A single item in the oracle dataset."""

    statement: str
    is_true: bool
    encoded: np.ndarray | None = None

    def encode(self, dim: int = 32) -> np.ndarray:
        if self.encoded is None:
            self.encoded = encode_statement(self.statement, dim)
        return self.encoded


class OracleDataset:
    """Ground-truth dataset for training the Phase Cancellation system."""

    def __init__(self, items: Sequence[OracleItem] | None = None) -> None:
        self.items: list[OracleItem] = list(items) if items else []

    def add(self, statement: str, is_true: bool) -> None:
        self.items.append(OracleItem(statement=statement, is_true=is_true))

    @property
    def true_items(self) -> list[OracleItem]:
        return [it for it in self.items if it.is_true]

    @property
    def false_items(self) -> list[OracleItem]:
        return [it for it in self.items if not it.is_true]

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def create_demo(cls) -> OracleDataset:
        """Create a small demonstration dataset with obvious truths/falsehoods."""
        ds = cls()
        # True statements
        ds.add("water is wet", True)
        ds.add("the sun is hot", True)
        ds.add("fire burns", True)
        ds.add("ice is cold", True)
        ds.add("birds can fly", True)
        ds.add("fish live in water", True)
        ds.add("trees have leaves", True)
        ds.add("the earth orbits the sun", True)
        ds.add("humans need oxygen", True)
        ds.add("gravity pulls things down", True)
        ds.add("light travels fast", True)
        ds.add("sound needs a medium", True)
        ds.add("metals conduct electricity", True)
        ds.add("plants need sunlight", True)
        ds.add("the moon orbits earth", True)

        # False statements
        ds.add("water is dry", False)
        ds.add("the sun is cold", False)
        ds.add("fire freezes", False)
        ds.add("ice is hot", False)
        ds.add("rocks can fly", False)
        ds.add("fish live in fire", False)
        ds.add("trees have wheels", False)
        ds.add("the earth orbits the moon", False)
        ds.add("humans need poison", False)
        ds.add("gravity pushes things up", False)
        ds.add("light stands still", False)
        ds.add("sound travels in vacuum", False)
        ds.add("wood conducts electricity", False)
        ds.add("plants need darkness", False)
        ds.add("the moon orbits mars", False)

        return ds


def fitness_embedder_excitatory(
    genome: Genome,
    embedder_i_genome: Genome,
    interference_genome: Genome | None,
    boundary_genome: Genome,
    oracle: OracleDataset,
    input_dim: int = 32,
    use_learned_interference: bool = False,
) -> float:
    """Fitness function for the Excitatory Embedder.

    Rewarded when:
    - True statements produce strong residual signals (survive cancellation)
    - Its signal is stronger than Embedder_I for true statements
    """
    net_e = FeedForwardNetwork(genome)
    net_i = FeedForwardNetwork(embedder_i_genome)
    net_boundary = FeedForwardNetwork(boundary_genome)
    net_interference = FeedForwardNetwork(interference_genome) if interference_genome else None

    score = 0.0

    for item in oracle.items:
        encoded = item.encode(input_dim)
        pos_e = net_e.activate(encoded)
        pos_i = net_i.activate(encoded)

        # Compute residual
        if net_interference is not None and use_learned_interference:
            combined = np.concatenate([pos_e, pos_i])
            residual = net_interference.activate(combined)
        else:
            residual = pos_e + pos_i

        truth_score = net_boundary.activate(residual)[0]
        signal_strength = np.linalg.norm(residual)

        if item.is_true:
            # True: want high truth_score AND strong signal
            score += truth_score * (1.0 + signal_strength)
        else:
            # False: excitatory embedder is NOT penalized for false items
            # (that's the inhibitory embedder's job)
            # But mild bonus if false items happen to score low
            score += (1.0 - truth_score) * 0.3

    return max(0.001, score / len(oracle))


def fitness_embedder_inhibitory(
    genome: Genome,
    embedder_e_genome: Genome,
    interference_genome: Genome | None,
    boundary_genome: Genome,
    oracle: OracleDataset,
    input_dim: int = 32,
    use_learned_interference: bool = False,
) -> float:
    """Fitness function for the Inhibitory Embedder.

    Rewarded when:
    - False statements are cancelled (weak residual, low truth score)
    - Its signal is stronger than Embedder_E for false statements
    """
    net_e = FeedForwardNetwork(embedder_e_genome)
    net_i = FeedForwardNetwork(genome)
    net_boundary = FeedForwardNetwork(boundary_genome)
    net_interference = FeedForwardNetwork(interference_genome) if interference_genome else None

    score = 0.0

    for item in oracle.items:
        encoded = item.encode(input_dim)
        pos_e = net_e.activate(encoded)
        pos_i = net_i.activate(encoded)

        if net_interference is not None and use_learned_interference:
            combined = np.concatenate([pos_e, pos_i])
            residual = net_interference.activate(combined)
        else:
            residual = pos_e + pos_i

        truth_score = net_boundary.activate(residual)[0]
        signal_strength = np.linalg.norm(residual)

        if not item.is_true:
            # False: want LOW truth_score AND weak signal (good cancellation)
            cancellation_quality = 1.0 / (1.0 + signal_strength)
            score += (1.0 - truth_score) * (1.0 + cancellation_quality)
        else:
            # True: inhibitory embedder should NOT cancel true statements
            # Mild bonus for letting true statements through
            score += truth_score * 0.3

    return max(0.001, score / len(oracle))


def fitness_interference(
    genome: Genome,
    embedder_e_genome: Genome,
    embedder_i_genome: Genome,
    boundary_genome: Genome,
    oracle: OracleDataset,
    input_dim: int = 32,
) -> float:
    """Fitness function for the Interference network.

    Rewarded when:
    1. True statements survive cancellation (strong residual, high boundary score)
    2. False statements are cancelled (weak residual, low boundary score)
    3. Confidence correlates with truth clarity
    """
    net_e = FeedForwardNetwork(embedder_e_genome)
    net_i = FeedForwardNetwork(embedder_i_genome)
    net_interference = FeedForwardNetwork(genome)
    net_boundary = FeedForwardNetwork(boundary_genome)

    score = 0.0
    true_strengths = []
    false_strengths = []

    for item in oracle.items:
        encoded = item.encode(input_dim)
        pos_e = net_e.activate(encoded)
        pos_i = net_i.activate(encoded)
        combined = np.concatenate([pos_e, pos_i])
        residual = net_interference.activate(combined)

        truth_score = net_boundary.activate(residual)[0]
        signal_strength = np.linalg.norm(residual)

        if item.is_true:
            score += truth_score * signal_strength
            true_strengths.append(signal_strength)
        else:
            cancellation_quality = 1.0 / (1.0 + signal_strength)
            score += (1.0 - truth_score) * cancellation_quality
            false_strengths.append(signal_strength)

    # Separation bonus: true signals should be stronger than false signals
    if true_strengths and false_strengths:
        mean_true = np.mean(true_strengths)
        mean_false = np.mean(false_strengths)
        if mean_true > mean_false:
            separation = (mean_true - mean_false) / (mean_true + mean_false + 1e-8)
            score += separation * len(oracle)

    return max(0.001, score / len(oracle))


def fitness_boundary(
    genome: Genome,
    embedder_e_genome: Genome,
    embedder_i_genome: Genome,
    interference_genome: Genome | None,
    oracle: OracleDataset,
    input_dim: int = 32,
    use_learned_interference: bool = False,
) -> float:
    """Fitness function for the Boundary network.

    Rewarded for correctly classifying residual positions as true/false.
    """
    net_e = FeedForwardNetwork(embedder_e_genome)
    net_i = FeedForwardNetwork(embedder_i_genome)
    net_boundary = FeedForwardNetwork(genome)
    net_interference = FeedForwardNetwork(interference_genome) if interference_genome else None

    score = 0.0

    for item in oracle.items:
        encoded = item.encode(input_dim)
        pos_e = net_e.activate(encoded)
        pos_i = net_i.activate(encoded)

        if net_interference is not None and use_learned_interference:
            combined = np.concatenate([pos_e, pos_i])
            residual = net_interference.activate(combined)
        else:
            residual = pos_e + pos_i

        truth_score = net_boundary.activate(residual)[0]

        if item.is_true:
            score += truth_score
        else:
            score += 1.0 - truth_score

    return max(0.001, score / len(oracle))
