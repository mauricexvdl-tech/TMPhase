"""NEAT genome representation.

A genome encodes a neural network topology as a collection of node genes
and connection genes. Mutations can add nodes, add connections, or perturb
weights. Crossover recombines two genomes based on matching innovation numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class NodeGene:
    """A node in the NEAT genome."""

    id: int
    type: str  # "input", "output", "hidden"
    activation: str = "tanh"
    bias: float = 0.0


@dataclass
class ConnectionGene:
    """A connection (edge) in the NEAT genome."""

    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0


class InnovationTracker:
    """Global tracker for structural innovation numbers."""

    def __init__(self) -> None:
        self._counter = 0
        self._history: dict[tuple[int, int], int] = {}

    def get_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._history:
            self._history[key] = self._counter
            self._counter += 1
        return self._history[key]

    @property
    def counter(self) -> int:
        return self._counter


# Module-level default tracker (shared within a run)
_global_tracker = InnovationTracker()


def get_global_tracker() -> InnovationTracker:
    return _global_tracker


def reset_global_tracker() -> None:
    global _global_tracker
    _global_tracker = InnovationTracker()


@dataclass
class Genome:
    """A NEAT genome encoding a neural network topology."""

    nodes: list[NodeGene] = field(default_factory=list)
    connections: list[ConnectionGene] = field(default_factory=list)
    fitness: float = 0.0
    _next_node_id: int = 0

    # --- Construction helpers ---

    @classmethod
    def create_minimal(
        cls,
        n_inputs: int,
        n_outputs: int,
        activation: str = "tanh",
        tracker: Optional[InnovationTracker] = None,
    ) -> Genome:
        """Create a minimal fully-connected genome (no hidden nodes)."""
        if tracker is None:
            tracker = get_global_tracker()

        genome = cls()
        # Input nodes
        for i in range(n_inputs):
            genome.nodes.append(NodeGene(id=i, type="input", activation="identity"))
        # Output nodes
        for i in range(n_outputs):
            nid = n_inputs + i
            genome.nodes.append(NodeGene(id=nid, type="output", activation=activation))
        genome._next_node_id = n_inputs + n_outputs

        # Full connections from inputs to outputs with small random weights
        for inp in range(n_inputs):
            for out in range(n_inputs, n_inputs + n_outputs):
                innov = tracker.get_innovation(inp, out)
                w = random.gauss(0.0, 1.0)
                genome.connections.append(
                    ConnectionGene(in_node=inp, out_node=out, weight=w, innovation=innov)
                )
        return genome

    # --- Queries ---

    @property
    def input_nodes(self) -> list[NodeGene]:
        return [n for n in self.nodes if n.type == "input"]

    @property
    def output_nodes(self) -> list[NodeGene]:
        return [n for n in self.nodes if n.type == "output"]

    @property
    def hidden_nodes(self) -> list[NodeGene]:
        return [n for n in self.nodes if n.type == "hidden"]

    def get_node(self, node_id: int) -> Optional[NodeGene]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    # --- Mutations ---

    def mutate(
        self,
        tracker: Optional[InnovationTracker] = None,
        weight_perturb_rate: float = 0.8,
        weight_replace_rate: float = 0.1,
        add_node_rate: float = 0.05,
        add_conn_rate: float = 0.08,
        toggle_rate: float = 0.01,
        bias_perturb_rate: float = 0.3,
        weight_perturb_power: float = 0.5,
        bias_perturb_power: float = 0.2,
    ) -> None:
        """Apply mutations to this genome in-place."""
        if tracker is None:
            tracker = get_global_tracker()

        # Weight mutations
        for conn in self.connections:
            if random.random() < weight_perturb_rate:
                if random.random() < weight_replace_rate:
                    conn.weight = random.gauss(0.0, 1.0)
                else:
                    conn.weight += random.gauss(0.0, weight_perturb_power)
                conn.weight = np.clip(conn.weight, -5.0, 5.0)

        # Bias mutations
        for node in self.nodes:
            if node.type != "input" and random.random() < bias_perturb_rate:
                node.bias += random.gauss(0.0, bias_perturb_power)
                node.bias = np.clip(node.bias, -5.0, 5.0)

        # Structural: add node
        if random.random() < add_node_rate:
            self._mutate_add_node(tracker)

        # Structural: add connection
        if random.random() < add_conn_rate:
            self._mutate_add_connection(tracker)

        # Toggle connection
        if self.connections and random.random() < toggle_rate:
            conn = random.choice(self.connections)
            conn.enabled = not conn.enabled

    def _mutate_add_node(self, tracker: InnovationTracker) -> None:
        """Split an existing connection by inserting a new hidden node."""
        enabled = [c for c in self.connections if c.enabled]
        if not enabled:
            return
        conn = random.choice(enabled)
        conn.enabled = False

        new_id = self._next_node_id
        self._next_node_id += 1

        activation = random.choice(["tanh", "relu", "sigmoid", "gauss", "sin"])
        self.nodes.append(NodeGene(id=new_id, type="hidden", activation=activation))

        innov_a = tracker.get_innovation(conn.in_node, new_id)
        innov_b = tracker.get_innovation(new_id, conn.out_node)
        self.connections.append(
            ConnectionGene(in_node=conn.in_node, out_node=new_id, weight=1.0, innovation=innov_a)
        )
        self.connections.append(
            ConnectionGene(
                in_node=new_id, out_node=conn.out_node, weight=conn.weight, innovation=innov_b
            )
        )

    def _mutate_add_connection(self, tracker: InnovationTracker) -> None:
        """Add a new connection between two previously unconnected nodes."""
        node_ids = [n.id for n in self.nodes]
        input_ids = {n.id for n in self.input_nodes}
        output_ids = {n.id for n in self.output_nodes}

        # Try a few times to find a valid new connection
        for _ in range(20):
            src = random.choice(node_ids)
            dst = random.choice(node_ids)
            # Don't connect to inputs, from outputs, or self-loops
            if dst in input_ids or src in output_ids or src == dst:
                continue
            # Check no duplicate
            exists = any(c.in_node == src and c.out_node == dst for c in self.connections)
            if exists:
                continue
            innov = tracker.get_innovation(src, dst)
            w = random.gauss(0.0, 1.0)
            self.connections.append(
                ConnectionGene(in_node=src, out_node=dst, weight=w, innovation=innov)
            )
            return

    # --- Crossover ---

    @staticmethod
    def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
        """NEAT crossover: matching genes inherited randomly, disjoint/excess from fitter parent."""
        if parent_b.fitness > parent_a.fitness:
            parent_a, parent_b = parent_b, parent_a

        child = Genome()

        # Build innovation -> connection maps
        genes_a = {c.innovation: c for c in parent_a.connections}
        genes_b = {c.innovation: c for c in parent_b.connections}

        all_innovations = set(genes_a.keys()) | set(genes_b.keys())
        for innov in sorted(all_innovations):
            if innov in genes_a and innov in genes_b:
                # Matching gene: inherit randomly
                gene = random.choice([genes_a[innov], genes_b[innov]])
            elif innov in genes_a:
                # Disjoint/excess from fitter parent
                gene = genes_a[innov]
            else:
                # Skip genes only in weaker parent
                continue
            child.connections.append(
                ConnectionGene(
                    in_node=gene.in_node,
                    out_node=gene.out_node,
                    weight=gene.weight,
                    enabled=gene.enabled,
                    innovation=gene.innovation,
                )
            )

        # Collect all node IDs referenced in child connections
        node_ids_needed = set()
        for c in child.connections:
            node_ids_needed.add(c.in_node)
            node_ids_needed.add(c.out_node)
        # Also include all input/output nodes from parent_a
        for n in parent_a.nodes:
            if n.type in ("input", "output"):
                node_ids_needed.add(n.id)

        node_map_a = {n.id: n for n in parent_a.nodes}
        node_map_b = {n.id: n for n in parent_b.nodes}
        for nid in sorted(node_ids_needed):
            if nid in node_map_a:
                src = node_map_a[nid]
            elif nid in node_map_b:
                src = node_map_b[nid]
            else:
                continue
            child.nodes.append(
                NodeGene(id=src.id, type=src.type, activation=src.activation, bias=src.bias)
            )

        child._next_node_id = max((n.id for n in child.nodes), default=0) + 1
        return child

    # --- Distance (for speciation) ---

    @staticmethod
    def distance(
        g1: Genome,
        g2: Genome,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
    ) -> float:
        """Compute compatibility distance between two genomes."""
        genes_1 = {c.innovation: c for c in g1.connections}
        genes_2 = {c.innovation: c for c in g2.connections}

        matching = set(genes_1.keys()) & set(genes_2.keys())
        disjoint_excess = len(set(genes_1.keys()) ^ set(genes_2.keys()))
        n = max(len(g1.connections), len(g2.connections), 1)

        weight_diff = 0.0
        if matching:
            weight_diff = sum(
                abs(genes_1[i].weight - genes_2[i].weight) for i in matching
            ) / len(matching)

        return (c1 * disjoint_excess / n) + (c3 * weight_diff)

    # --- Copy ---

    def copy(self) -> Genome:
        g = Genome()
        g.nodes = [
            NodeGene(id=n.id, type=n.type, activation=n.activation, bias=n.bias)
            for n in self.nodes
        ]
        g.connections = [
            ConnectionGene(
                in_node=c.in_node,
                out_node=c.out_node,
                weight=c.weight,
                enabled=c.enabled,
                innovation=c.innovation,
            )
            for c in self.connections
        ]
        g.fitness = self.fitness
        g._next_node_id = self._next_node_id
        return g
