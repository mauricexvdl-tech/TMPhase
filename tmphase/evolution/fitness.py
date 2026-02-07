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


def _complexity_penalty(genome: Genome, weight: float = 0.001) -> float:
    """Penalize genome complexity to prevent overfitting.

    Counts enabled connections and hidden nodes — simpler networks
    that achieve the same fitness are preferred (Occam's razor).
    """
    n_connections = sum(1 for c in genome.connections if c.enabled)
    n_hidden = len(genome.hidden_nodes)
    return weight * (n_connections + 2.0 * n_hidden)


@dataclass
class OracleItem:
    """A single item in the oracle dataset."""

    statement: str
    is_true: bool
    _encoded: np.ndarray | None = None
    _encoded_dim: int = 0

    def encode(self, dim: int = 64) -> np.ndarray:
        if self._encoded is None or self._encoded_dim != dim:
            self._encoded = encode_statement(self.statement, dim)
            self._encoded_dim = dim
        return self._encoded


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


def fitness_warmup(
    genome: Genome,
    boundary_genome: Genome,
    oracle: OracleDataset,
    input_dim: int = 48,
    is_boundary: bool = False,
    complexity_weight: float = 0.001,
) -> float:
    """Warm-up fitness: single-path (E only → Boundary), no I at all.

    Used during the warm-up phase so E and Boundary can learn basic
    classification before adversarial pressure from I.
    If is_boundary=True, the genome is the boundary; otherwise it's E.
    """
    if is_boundary:
        net_e = FeedForwardNetwork(boundary_genome)  # boundary_genome is actually the E genome
        net_b = FeedForwardNetwork(genome)
    else:
        net_e = FeedForwardNetwork(genome)
        net_b = FeedForwardNetwork(boundary_genome)

    score = 0.0
    for item in oracle.items:
        encoded = item.encode(input_dim)
        pos_e = net_e.activate(encoded)
        truth_score = net_b.activate(pos_e)[0]
        if item.is_true:
            score += truth_score
        else:
            score += 1.0 - truth_score
    raw = score / len(oracle)
    return max(0.001, raw - _complexity_penalty(genome, complexity_weight))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, returns 0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def fitness_embedder_excitatory(
    genome: Genome,
    embedder_i_genome: Genome,
    interference_genome: Genome | None,
    boundary_genome: Genome,
    oracle: OracleDataset,
    input_dim: int = 64,
    use_learned_interference: bool = False,
    complexity_weight: float = 0.001,
) -> float:
    """Fitness function for the Excitatory Embedder.

    Rewarded when:
    - True statements produce strong residual signals (survive cancellation)
    - E and I vectors are aligned for true statements (constructive interference)
    - E and I vectors are anti-aligned for false statements (destructive interference)
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
        cos_ei = _cosine_similarity(pos_e, pos_i)

        if item.is_true:
            # True: want high truth_score, strong signal, E aligned with I
            alignment_bonus = max(0.0, cos_ei)  # reward alignment
            score += truth_score * (1.0 + signal_strength) + alignment_bonus * 0.5
        else:
            # False: reward if E cooperates with I to cancel
            anti_alignment_bonus = max(0.0, -cos_ei)  # reward anti-alignment
            score += (1.0 - truth_score) * 0.5 + anti_alignment_bonus * 0.3

    raw = score / len(oracle)
    return max(0.001, raw - _complexity_penalty(genome, complexity_weight))


def fitness_embedder_inhibitory(
    genome: Genome,
    embedder_e_genome: Genome,
    interference_genome: Genome | None,
    boundary_genome: Genome,
    oracle: OracleDataset,
    input_dim: int = 64,
    use_learned_interference: bool = False,
    complexity_weight: float = 0.001,
) -> float:
    """Fitness function for the Inhibitory Embedder.

    Rewarded when:
    - False statements: I produces vectors anti-aligned with E (destructive interference)
    - True statements: I produces vectors aligned with E (constructive, lets truth through)
    - This creates the core phase cancellation dynamic.
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
        cos_ei = _cosine_similarity(pos_e, pos_i)

        if not item.is_true:
            # False: want anti-aligned vectors (destructive interference)
            cancellation_quality = 1.0 / (1.0 + signal_strength)
            anti_alignment = max(0.0, -cos_ei)  # reward negative cosine
            score += (1.0 - truth_score) * (1.0 + cancellation_quality) + anti_alignment * 1.0
        else:
            # True: I should be aligned with E (constructive interference)
            alignment = max(0.0, cos_ei)  # reward positive cosine
            score += truth_score * 0.5 + alignment * 0.5

    raw = score / len(oracle)
    return max(0.001, raw - _complexity_penalty(genome, complexity_weight))


def fitness_interference(
    genome: Genome,
    embedder_e_genome: Genome,
    embedder_i_genome: Genome,
    boundary_genome: Genome,
    oracle: OracleDataset,
    input_dim: int = 64,
    complexity_weight: float = 0.001,
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
            score += separation * len(oracle) * 2.0  # doubled separation reward

    raw = score / len(oracle)
    return max(0.001, raw - _complexity_penalty(genome, complexity_weight))


def fitness_boundary(
    genome: Genome,
    embedder_e_genome: Genome,
    embedder_i_genome: Genome,
    interference_genome: Genome | None,
    oracle: OracleDataset,
    input_dim: int = 64,
    use_learned_interference: bool = False,
    complexity_weight: float = 0.001,
) -> float:
    """Fitness function for the Boundary network.

    Rewarded for correctly classifying residual positions as true/false.
    Uses linear reward + bonus for confident CORRECT predictions.
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
            # Bonus only for confident CORRECT predictions
            if truth_score > 0.6:
                score += (truth_score - 0.6) * 0.5
        else:
            score += 1.0 - truth_score
            if truth_score < 0.4:
                score += (0.4 - truth_score) * 0.5

    raw = score / len(oracle)
    return max(0.001, raw - _complexity_penalty(genome, complexity_weight))
