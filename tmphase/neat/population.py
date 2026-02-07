"""NEAT population management — selection, reproduction, and evolution."""

from __future__ import annotations

import random
from typing import Callable, Optional

from .genome import Genome, InnovationTracker, get_global_tracker
from .species import Species, speciate


class Population:
    """A population of NEAT genomes with speciation and selection."""

    def __init__(
        self,
        size: int,
        n_inputs: int,
        n_outputs: int,
        output_activation: str = "tanh",
        tracker: Optional[InnovationTracker] = None,
    ) -> None:
        self.size = size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tracker = tracker or get_global_tracker()
        self.generation = 0
        self.species: list[Species] = []
        self.best_genome: Optional[Genome] = None

        # Create initial population
        self.genomes: list[Genome] = []
        for _ in range(size):
            g = Genome.create_minimal(n_inputs, n_outputs, output_activation, self.tracker)
            g.mutate(self.tracker)
            self.genomes.append(g)

    def evaluate(self, fitness_fn: Callable[[Genome], float]) -> None:
        """Evaluate all genomes using the provided fitness function."""
        for genome in self.genomes:
            genome.fitness = fitness_fn(genome)

        best = max(self.genomes, key=lambda g: g.fitness)
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best.copy()

    def evolve(
        self,
        elitism: int = 2,
        survival_rate: float = 0.2,
        crossover_rate: float = 0.75,
        compatibility_threshold: float = 3.0,
        stagnation_limit: int = 15,
    ) -> None:
        """Run one generation of evolution."""
        # Speciation
        self.species = speciate(self.genomes, self.species, compatibility_threshold)

        # Remove stagnant species (keep at least one)
        if len(self.species) > 1:
            self.species = [
                sp for sp in self.species if sp.stagnation < stagnation_limit
            ] or [max(self.species, key=lambda s: s.best_fitness)]

        # Compute adjusted fitness and offspring allocation
        total_adj_fitness = 0.0
        for sp in self.species:
            for g in sp.members:
                g.fitness = max(g.fitness, 0.001)
            sp_adj = sum(g.fitness / len(sp.members) for g in sp.members)
            total_adj_fitness += sp_adj

        new_genomes: list[Genome] = []

        # Elitism: keep best from each species
        for sp in self.species:
            sp.members.sort(key=lambda g: g.fitness, reverse=True)
            for g in sp.members[:elitism]:
                if len(new_genomes) < self.size:
                    new_genomes.append(g.copy())

        # Produce offspring
        for sp in self.species:
            if not sp.members:
                continue
            sp_adj = sum(g.fitness / len(sp.members) for g in sp.members)
            n_offspring = max(
                1,
                int(round(sp_adj / max(total_adj_fitness, 1e-8) * self.size)) - elitism,
            )

            # Select parents from top fraction
            cutoff = max(1, int(len(sp.members) * survival_rate))
            parents = sp.members[:cutoff]

            for _ in range(n_offspring):
                if len(new_genomes) >= self.size:
                    break
                if len(parents) > 1 and random.random() < crossover_rate:
                    p1, p2 = random.sample(parents, 2)
                    child = Genome.crossover(p1, p2)
                else:
                    child = random.choice(parents).copy()
                child.mutate(self.tracker)
                new_genomes.append(child)

        # Fill remaining slots if needed
        while len(new_genomes) < self.size:
            g = random.choice(self.genomes).copy()
            g.mutate(self.tracker)
            new_genomes.append(g)

        self.genomes = new_genomes[: self.size]
        self.generation += 1
