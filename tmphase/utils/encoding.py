"""Encoding utilities for converting symbolic inputs to numeric vectors.

Two encoding strategies:
1. Hash-based (original): deterministic but semantically blind
2. N-gram based (new): similar words produce similar vectors
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


# ── Hash-based encoding (original) ─────────────────────────────────

def encode_symbol(symbol: str, dim: int = 32) -> np.ndarray:
    """Encode a symbolic string into a fixed-dimension numeric vector.

    Uses a deterministic hash-based encoding so the same symbol always
    produces the same vector. Values are in [-1, 1].
    """
    h = hashlib.sha256(symbol.encode("utf-8")).digest()
    # Use enough bytes to fill the requested dimension
    while len(h) < dim:
        h += hashlib.sha256(h).digest()
    # Convert bytes to uint8 then scale to [-1, 1] (no NaN/Inf possible)
    raw = np.frombuffer(h[:dim], dtype=np.uint8).astype(np.float32)
    raw = (raw / 127.5) - 1.0  # Maps [0, 255] -> [-1.0, ~1.0]
    return raw[:dim]


# ── N-gram based encoding (semantic) ───────────────────────────────

def _char_ngrams(word: str, n: int = 3) -> list[str]:
    """Extract character n-grams from a word, including boundary markers."""
    padded = f"#{word}#"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _ngram_to_vec(ngram: str, dim: int) -> np.ndarray:
    """Map a single n-gram to a deterministic vector using hashing trick."""
    h = hashlib.md5(ngram.encode("utf-8")).digest()
    while len(h) < dim:
        h += hashlib.md5(h).digest()
    raw = np.frombuffer(h[:dim], dtype=np.uint8).astype(np.float32)
    return (raw / 127.5) - 1.0


def encode_word_ngram(word: str, dim: int = 32) -> np.ndarray:
    """Encode a word using character n-gram averaging.

    Key property: similar words (e.g. "hot" / "not", "wet" / "set")
    share some n-grams and thus produce partially similar vectors,
    while very different words produce orthogonal vectors.
    """
    word = word.lower().strip()
    if not word:
        return np.zeros(dim, dtype=np.float32)

    # Use 2-grams and 3-grams for a richer representation
    ngrams = _char_ngrams(word, 2) + _char_ngrams(word, 3)
    if not ngrams:
        return encode_symbol(word, dim)

    vecs = [_ngram_to_vec(ng, dim) for ng in ngrams]
    combined = np.mean(vecs, axis=0)
    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined = combined / norm
    return combined.astype(np.float32)


def encode_statement_ngram(statement: str, dim: int = 32) -> np.ndarray:
    """Encode a statement using n-gram word embeddings.

    Words are encoded individually with n-grams, then combined using
    position-weighted averaging (earlier words weighted slightly more,
    giving rudimentary word-order sensitivity).
    """
    words = statement.lower().split()
    if not words:
        return np.zeros(dim, dtype=np.float32)

    vectors = []
    weights = []
    for i, word in enumerate(words):
        vectors.append(encode_word_ngram(word, dim))
        # Position weight: mild decay so word order matters slightly
        weights.append(1.0 / (1.0 + 0.1 * i))

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    combined = np.zeros(dim, dtype=np.float32)
    for v, w in zip(vectors, weights):
        combined += w * v

    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined = combined / norm
    return combined.astype(np.float32)


# ── Unified interface ──────────────────────────────────────────────

def encode_statement(statement: str, dim: int = 32, method: str = "ngram") -> np.ndarray:
    """Encode a statement into a fixed-dimension vector.

    Args:
        method: "hash" for original hash-based, "ngram" for semantic n-gram
    """
    if method == "hash":
        words = statement.lower().split()
        if not words:
            return np.zeros(dim, dtype=np.float32)
        vectors = [encode_symbol(w, dim) for w in words]
        combined = np.mean(vectors, axis=0)
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined = combined / norm
        return combined.astype(np.float32)
    else:
        return encode_statement_ngram(statement, dim)


def encode_batch(statements: Sequence[str], dim: int = 32, method: str = "ngram") -> np.ndarray:
    """Encode a batch of statements. Returns shape (N, dim)."""
    return np.array([encode_statement(s, dim, method) for s in statements], dtype=np.float32)
