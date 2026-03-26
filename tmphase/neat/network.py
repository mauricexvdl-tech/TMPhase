"""Feed-forward neural network built from a NEAT genome."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .genome import Genome, NodeGene
from ..utils.activations import get_activation


class FeedForwardNetwork:
    """A feed-forward network decoded from a NEAT genome.

    Evaluates nodes in topologically sorted order (no recurrent connections).
    """

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self._input_ids = [n.id for n in genome.input_nodes]
        self._output_ids = [n.id for n in genome.output_nodes]

        # Build adjacency and node info
        self._node_map: dict[int, NodeGene] = {n.id: n for n in genome.nodes}
        self._incoming: dict[int, list[tuple[int, float]]] = {n.id: [] for n in genome.nodes}
        for conn in genome.connections:
            if conn.enabled and conn.out_node in self._incoming and conn.in_node in self._node_map:
                self._incoming[conn.out_node].append((conn.in_node, conn.weight))

        # Topological sort (Kahn's algorithm)
        self._eval_order = self._topological_sort()

    def _topological_sort(self) -> list[int]:
        """Return node IDs in evaluation order (inputs first, then hidden/output)."""
        in_degree: dict[int, int] = {n.id: 0 for n in self.genome.nodes}
        adjacency: dict[int, list[int]] = {n.id: [] for n in self.genome.nodes}
        for conn in self.genome.connections:
            if conn.enabled and conn.out_node in in_degree and conn.in_node in in_degree:
                in_degree[conn.out_node] += 1
                adjacency[conn.in_node].append(conn.out_node)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for neighbor in adjacency.get(nid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If there are cycles, append remaining nodes (recurrent connections ignored)
        visited = set(order)
        for n in self.genome.nodes:
            if n.id not in visited:
                order.append(n.id)
        return order

    def activate(self, inputs: np.ndarray | Sequence[float]) -> np.ndarray:
        """Run a forward pass. Returns output node values as a numpy array."""
        inputs = np.asarray(inputs, dtype=np.float64)
        if len(inputs) != len(self._input_ids):
            raise ValueError(
                f"Expected {len(self._input_ids)} inputs, got {len(inputs)}"
            )

        values: dict[int, float] = {}
        # Set input values
        for i, nid in enumerate(self._input_ids):
            values[nid] = float(inputs[i])

        # Evaluate in topological order
        for nid in self._eval_order:
            if nid in values and self._node_map[nid].type == "input":
                continue  # Already set

            node = self._node_map[nid]
            incoming = self._incoming.get(nid, [])
            if incoming:
                s = sum(values.get(src, 0.0) * w for src, w in incoming)
            else:
                s = 0.0
            s += node.bias

            act_fn = get_activation(node.activation)
            values[nid] = float(act_fn(np.array([s]))[0])

        return np.array([values.get(nid, 0.0) for nid in self._output_ids], dtype=np.float64)
