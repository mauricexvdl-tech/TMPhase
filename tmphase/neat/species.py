"""Species management for NEAT speciation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .genome import Genome


@dataclass
class Species:
    """A species groups genomes that are topologically similar."""

    id: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = 0.0
    stagnation: int = 0

    def update_representative(self) -> None:
        if self.members:
            self.representative = random.choice(self.members)

    def update_stagnation(self) -> None:
        current_best = max((g.fitness for g in self.members), default=0.0)
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.stagnation = 0
        else:
            self.stagnation += 1


def speciate(
    genomes: list[Genome],
    existing_species: list[Species],
    threshold: float = 3.0,
) -> list[Species]:
    """Assign genomes to species based on compatibility distance."""
    # Clear old members
    for sp in existing_species:
        sp.members = []

    next_id = max((sp.id for sp in existing_species), default=0) + 1

    for genome in genomes:
        placed = False
        for sp in existing_species:
            dist = Genome.distance(genome, sp.representative)
            if dist < threshold:
                sp.members.append(genome)
                placed = True
                break
        if not placed:
            new_sp = Species(id=next_id, representative=genome, members=[genome])
            existing_species.append(new_sp)
            next_id += 1

    # Remove empty species
    existing_species = [sp for sp in existing_species if sp.members]

    # Update representatives and stagnation
    for sp in existing_species:
        sp.update_stagnation()
        sp.update_representative()

    return existing_species
