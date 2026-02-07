"""Tests for the co-evolutionary training pipeline."""

from tmphase.evolution.fitness import OracleDataset, OracleItem
from tmphase.evolution.coevolution import PhaseCancellationSystem, SystemConfig


class TestOracleDataset:
    def test_create_demo(self):
        oracle = OracleDataset.create_demo()
        assert len(oracle) == 30
        assert len(oracle.true_items) == 15
        assert len(oracle.false_items) == 15

    def test_add(self):
        oracle = OracleDataset()
        oracle.add("test statement", True)
        assert len(oracle) == 1
        assert oracle.items[0].is_true is True

    def test_encode_cached(self):
        item = OracleItem(statement="hello world", is_true=True)
        enc1 = item.encode(dim=16)
        enc2 = item.encode(dim=16)
        assert enc1 is enc2  # Same object (cached)


class TestPhaseCancellationSystem:
    def test_creation(self):
        config = SystemConfig(population_size=10, input_dim=16, position_dim=8)
        system = PhaseCancellationSystem(config=config)
        assert system.generation == 0

    def test_single_generation(self):
        config = SystemConfig(population_size=10, input_dim=16, position_dim=8)
        oracle = OracleDataset()
        oracle.add("water is wet", True)
        oracle.add("water is dry", False)
        oracle.add("fire burns", True)
        oracle.add("fire freezes", False)

        system = PhaseCancellationSystem(config=config, oracle=oracle)
        stats = system.evolve_generation()

        assert stats.generation == 0
        assert 0.0 <= stats.accuracy <= 1.0
        assert stats.mean_true_signal >= 0.0
        assert stats.mean_false_signal >= 0.0

    def test_evaluate_statement(self):
        config = SystemConfig(population_size=10, input_dim=16, position_dim=8)
        oracle = OracleDataset()
        oracle.add("water is wet", True)
        oracle.add("water is dry", False)

        system = PhaseCancellationSystem(config=config, oracle=oracle)
        # Run one generation so we have best genomes
        system.evolve_generation()

        result = system.evaluate_statement("water is wet")
        assert "truth_score" in result
        assert "signal_strength" in result
        assert "confidence" in result
        assert "predicted_true" in result
        assert isinstance(result["predicted_true"], bool)

    def test_train_multiple_generations(self):
        config = SystemConfig(population_size=10, input_dim=16, position_dim=8)
        oracle = OracleDataset()
        oracle.add("sun is hot", True)
        oracle.add("sun is cold", False)

        system = PhaseCancellationSystem(config=config, oracle=oracle)
        stats = system.train(generations=3, verbose=False)

        assert len(stats) == 3
        assert system.generation == 3

    def test_additive_interference_mode(self):
        config = SystemConfig(
            population_size=10,
            input_dim=16,
            position_dim=8,
            use_learned_interference=False,
        )
        oracle = OracleDataset()
        oracle.add("fire burns", True)
        oracle.add("fire freezes", False)

        system = PhaseCancellationSystem(config=config, oracle=oracle)
        stats = system.evolve_generation()
        assert 0.0 <= stats.accuracy <= 1.0
