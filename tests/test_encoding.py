"""Tests for encoding utilities."""

import numpy as np

from tmphase.utils.encoding import encode_symbol, encode_statement, encode_batch


class TestEncodeSymbol:
    def test_returns_correct_dimension(self):
        vec = encode_symbol("hello", dim=32)
        assert vec.shape == (32,)

    def test_deterministic(self):
        a = encode_symbol("test", dim=16)
        b = encode_symbol("test", dim=16)
        np.testing.assert_array_equal(a, b)

    def test_different_symbols_differ(self):
        a = encode_symbol("cat", dim=32)
        b = encode_symbol("dog", dim=32)
        assert not np.allclose(a, b)

    def test_values_in_range(self):
        vec = encode_symbol("anything", dim=64)
        assert np.all(vec >= -1.0) and np.all(vec <= 1.0)


class TestEncodeStatement:
    def test_returns_correct_dimension(self):
        vec = encode_statement("the cat sat on the mat", dim=32)
        assert vec.shape == (32,)

    def test_empty_string_returns_zeros(self):
        vec = encode_statement("", dim=16)
        np.testing.assert_array_equal(vec, np.zeros(16))

    def test_normalized(self):
        vec = encode_statement("hello world", dim=32)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01 or norm < 0.01


class TestEncodeBatch:
    def test_batch_shape(self):
        stmts = ["hello", "world", "test"]
        batch = encode_batch(stmts, dim=32)
        assert batch.shape == (3, 32)

    def test_batch_matches_individual(self):
        stmts = ["alpha", "beta"]
        batch = encode_batch(stmts, dim=16)
        for i, s in enumerate(stmts):
            individual = encode_statement(s, dim=16)
            np.testing.assert_allclose(batch[i], individual, atol=1e-6)
