"""Tests for Phase Cancellation network components."""

import numpy as np

from tmphase.neat.genome import InnovationTracker
from tmphase.networks.embedder import Embedder, EmbedderConfig
from tmphase.networks.interference import (
    Interference,
    InterferenceConfig,
    InterferenceMode,
    additive_interference,
)
from tmphase.networks.boundary import Boundary, BoundaryConfig
from tmphase.networks.dynamics import Dynamics, DynamicsConfig
from tmphase.networks.pragmatics import (
    Reachability,
    ReachabilityConfig,
    Utility,
    UtilityConfig,
    Planner,
    PlannerConfig,
)
from tmphase.utils.encoding import encode_statement


class TestEmbedder:
    def test_excitatory_creation(self):
        tracker = InnovationTracker()
        emb = Embedder(EmbedderConfig(input_dim=32, output_dim=16), role="excitatory", tracker=tracker)
        assert emb.role == "excitatory"
        assert len(emb.population.genomes) == 100

    def test_embed_returns_correct_dim(self):
        tracker = InnovationTracker()
        emb = Embedder(
            EmbedderConfig(input_dim=32, output_dim=16, population_size=10),
            tracker=tracker,
        )
        encoded = encode_statement("test statement", dim=32)
        # Use a specific genome
        result = emb.embed_with_genome(emb.population.genomes[0], encoded)
        assert result.shape == (16,)


class TestInterference:
    def test_additive_interference(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        result = additive_interference(a, b)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_additive_mode(self):
        tracker = InnovationTracker()
        intf = Interference(
            InterferenceConfig(position_dim=4, output_dim=4),
            mode=InterferenceMode.ADDITIVE,
            tracker=tracker,
        )
        pos_e = np.array([1.0, 0.5, -0.3, 0.8])
        pos_i = np.array([-0.5, -0.5, 0.3, -0.8])
        result = intf.interfere(pos_e, pos_i)
        expected = pos_e + pos_i
        np.testing.assert_array_almost_equal(result, expected)

    def test_learned_mode_returns_correct_dim(self):
        tracker = InnovationTracker()
        intf = Interference(
            InterferenceConfig(position_dim=4, output_dim=4, population_size=10),
            mode=InterferenceMode.LEARNED,
            tracker=tracker,
        )
        pos_e = np.random.randn(4)
        pos_i = np.random.randn(4)
        result = intf.interfere_with_genome(
            intf.population.genomes[0], pos_e, pos_i
        )
        assert result.shape == (4,)

    def test_signal_strength(self):
        tracker = InnovationTracker()
        intf = Interference(
            InterferenceConfig(position_dim=4),
            mode=InterferenceMode.ADDITIVE,
            tracker=tracker,
        )
        residual = np.array([3.0, 4.0, 0.0, 0.0])
        assert abs(intf.signal_strength(residual) - 5.0) < 1e-6

    def test_confidence_constructive(self):
        tracker = InnovationTracker()
        intf = Interference(
            InterferenceConfig(position_dim=2),
            mode=InterferenceMode.ADDITIVE,
            tracker=tracker,
        )
        pos_e = np.array([1.0, 0.0])
        pos_i = np.array([1.0, 0.0])  # Same direction = constructive
        residual = pos_e + pos_i
        conf = intf.confidence(pos_e, pos_i, residual)
        assert conf > 1.0  # Constructive -> amplified

    def test_confidence_destructive(self):
        tracker = InnovationTracker()
        intf = Interference(
            InterferenceConfig(position_dim=2),
            mode=InterferenceMode.ADDITIVE,
            tracker=tracker,
        )
        pos_e = np.array([1.0, 0.0])
        pos_i = np.array([-1.0, 0.0])  # Opposite = destructive
        residual = pos_e + pos_i
        conf = intf.confidence(pos_e, pos_i, residual)
        assert conf < 0.1  # Destructive -> near zero


class TestBoundary:
    def test_evaluate_returns_scalar(self):
        tracker = InnovationTracker()
        bnd = Boundary(BoundaryConfig(input_dim=16, population_size=10), tracker=tracker)
        pos = np.random.randn(16)
        score = bnd.evaluate_with_genome(bnd.population.genomes[0], pos)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0  # Sigmoid output


class TestDynamics:
    def test_forward_inference(self):
        tracker = InnovationTracker()
        dyn = Dynamics(
            DynamicsConfig(position_dim=8, output_dim=8, population_size=10),
            role="forward",
            tracker=tracker,
        )
        a = np.random.randn(8)
        b = np.random.randn(8)
        result = dyn.infer_with_genome(dyn.population.genomes[0], a, b)
        assert result.shape == (8,)

    def test_contrary_inference(self):
        tracker = InnovationTracker()
        dyn = Dynamics(
            DynamicsConfig(position_dim=8, output_dim=8, population_size=10),
            role="contrary",
            tracker=tracker,
        )
        a = np.random.randn(8)
        b = np.random.randn(8)
        result = dyn.infer_with_genome(dyn.population.genomes[0], a, b)
        assert result.shape == (8,)


class TestReachability:
    def test_returns_correct_count(self):
        tracker = InnovationTracker()
        reach = Reachability(
            ReachabilityConfig(input_dim=8, n_reachable=3, position_dim=8, population_size=10),
            tracker=tracker,
        )
        # Need to set a best genome first
        from tmphase.neat.network import FeedForwardNetwork

        reach.population.best_genome = reach.population.genomes[0]
        positions = reach.get_reachable(np.random.randn(8))
        assert len(positions) == 3
        for p in positions:
            assert p.shape == (8,)


class TestUtility:
    def test_returns_bounded_scalar(self):
        tracker = InnovationTracker()
        util = Utility(
            UtilityConfig(input_dim=16, population_size=10),
            tracker=tracker,
        )
        util.population.best_genome = util.population.genomes[0]
        pos = np.random.randn(8)
        ctx = np.random.randn(8)
        score = util.evaluate(pos, ctx)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestPlanner:
    def test_returns_correct_dim(self):
        tracker = InnovationTracker()
        planner = Planner(
            PlannerConfig(input_dim=16, output_dim=8, population_size=10),
            tracker=tracker,
        )
        planner.population.best_genome = planner.population.genomes[0]
        current = np.random.randn(8)
        goal = np.random.randn(8)
        step = planner.next_step(current, goal)
        assert step.shape == (8,)
