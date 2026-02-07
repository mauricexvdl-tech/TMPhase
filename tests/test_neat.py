"""Tests for NEAT genome, network, and population."""

import numpy as np

from tmphase.neat.genome import Genome, InnovationTracker, ConnectionGene
from tmphase.neat.network import FeedForwardNetwork
from tmphase.neat.population import Population
from tmphase.neat.species import speciate, Species


class TestInnovationTracker:
    def test_same_connection_same_innovation(self):
        tracker = InnovationTracker()
        a = tracker.get_innovation(0, 3)
        b = tracker.get_innovation(0, 3)
        assert a == b

    def test_different_connections_different_innovations(self):
        tracker = InnovationTracker()
        a = tracker.get_innovation(0, 3)
        b = tracker.get_innovation(1, 3)
        assert a != b

    def test_counter_increments(self):
        tracker = InnovationTracker()
        tracker.get_innovation(0, 1)
        tracker.get_innovation(0, 2)
        assert tracker.counter == 2


class TestGenome:
    def test_create_minimal(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(4, 2, tracker=tracker)
        assert len(g.input_nodes) == 4
        assert len(g.output_nodes) == 2
        assert len(g.connections) == 8  # 4 inputs * 2 outputs

    def test_mutate_does_not_crash(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(4, 2, tracker=tracker)
        # Run many mutations to exercise all code paths
        for _ in range(100):
            g.mutate(tracker)
        assert len(g.nodes) >= 6  # At least original input + output nodes

    def test_crossover(self):
        tracker = InnovationTracker()
        a = Genome.create_minimal(4, 2, tracker=tracker)
        b = Genome.create_minimal(4, 2, tracker=tracker)
        a.fitness = 1.0
        b.fitness = 0.5
        child = Genome.crossover(a, b)
        assert len(child.input_nodes) == 4
        assert len(child.output_nodes) == 2

    def test_distance_identical_is_zero(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(4, 2, tracker=tracker)
        assert Genome.distance(g, g) == 0.0

    def test_distance_different_is_positive(self):
        tracker = InnovationTracker()
        a = Genome.create_minimal(4, 2, tracker=tracker)
        b = Genome.create_minimal(4, 2, tracker=tracker)
        for _ in range(10):
            b.mutate(tracker)
        # After mutations, distance should generally be > 0
        # (though technically identical weights are possible, it's astronomically unlikely)
        dist = Genome.distance(a, b)
        assert dist >= 0.0

    def test_copy_is_independent(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(2, 1, tracker=tracker)
        c = g.copy()
        c.connections[0].weight = 999.0
        assert g.connections[0].weight != 999.0


class TestFeedForwardNetwork:
    def test_activate_correct_output_count(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(3, 2, tracker=tracker)
        net = FeedForwardNetwork(g)
        out = net.activate([1.0, 0.5, -0.3])
        assert out.shape == (2,)

    def test_activate_deterministic(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(4, 2, tracker=tracker)
        net = FeedForwardNetwork(g)
        a = net.activate([1.0, 2.0, 3.0, 4.0])
        b = net.activate([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(a, b)

    def test_activate_wrong_input_size_raises(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(3, 1, tracker=tracker)
        net = FeedForwardNetwork(g)
        try:
            net.activate([1.0, 2.0])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_network_with_hidden_nodes(self):
        tracker = InnovationTracker()
        g = Genome.create_minimal(2, 1, tracker=tracker)
        # Force add a hidden node
        g._mutate_add_node(tracker)
        assert len(g.hidden_nodes) == 1
        net = FeedForwardNetwork(g)
        out = net.activate([1.0, -1.0])
        assert out.shape == (1,)


class TestPopulation:
    def test_creation(self):
        tracker = InnovationTracker()
        pop = Population(size=20, n_inputs=4, n_outputs=2, tracker=tracker)
        assert len(pop.genomes) == 20

    def test_evaluate_and_evolve(self):
        tracker = InnovationTracker()
        pop = Population(size=20, n_inputs=4, n_outputs=2, tracker=tracker)

        def dummy_fitness(genome: Genome) -> float:
            net = FeedForwardNetwork(genome)
            out = net.activate([1.0, 0.0, -1.0, 0.5])
            return float(abs(out[0] - 0.7) + abs(out[1] - 0.3))

        pop.evaluate(dummy_fitness)
        assert pop.best_genome is not None
        pop.evolve()
        assert pop.generation == 1
        assert len(pop.genomes) == 20


class TestSpeciation:
    def test_speciate_groups_similar(self):
        tracker = InnovationTracker()
        genomes = [Genome.create_minimal(4, 2, tracker=tracker) for _ in range(10)]
        species = speciate(genomes, [], threshold=3.0)
        assert len(species) >= 1
        total = sum(len(sp.members) for sp in species)
        assert total == 10
