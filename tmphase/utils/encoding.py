"""Encoding utilities for converting symbolic inputs to numeric vectors.

Three encoding strategies:
1. Hash-based (original): deterministic but semantically blind
2. N-gram based: similar words produce similar vectors
3. Bag-of-words hashed (BOW): each word activates specific dimensions,
   making minimal word differences maximally discriminative
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
    while len(h) < dim:
        h += hashlib.sha256(h).digest()
    raw = np.frombuffer(h[:dim], dtype=np.uint8).astype(np.float32)
    raw = (raw / 127.5) - 1.0
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
    """Encode a word using character n-gram averaging."""
    word = word.lower().strip()
    if not word:
        return np.zeros(dim, dtype=np.float32)
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
    """Encode a statement using n-gram word embeddings with position weighting."""
    words = statement.lower().split()
    if not words:
        return np.zeros(dim, dtype=np.float32)
    vectors = []
    weights = []
    for i, word in enumerate(words):
        vectors.append(encode_word_ngram(word, dim))
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


# ── Bag-of-words hashed encoding ───────────────────────────────────

def _word_hash_index(word: str, dim: int) -> int:
    """Map a word to a specific dimension index via hashing."""
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
    return h % dim


def _word_hash_sign(word: str) -> float:
    """Map a word to +1 or -1 (random sign hashing for variance reduction)."""
    h = int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16)
    return 1.0 if (h % 2 == 0) else -1.0


def _bigram_hash_index(bigram: str, dim: int) -> int:
    """Map a word bigram to a dimension index."""
    h = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
    return h % dim


def encode_statement_bow(statement: str, dim: int = 32) -> np.ndarray:
    """Encode using hashed bag-of-words + bigrams.

    Each word and word-bigram activates specific dimensions.
    Key advantage: "water boils at hundred degrees" vs
    "water boils at fifty degrees" differ in exactly the dimensions
    where "hundred" and "fifty" hash to. This makes small word
    differences maximally visible to the network.
    """
    words = statement.lower().split()
    if not words:
        return np.zeros(dim, dtype=np.float32)

    vec = np.zeros(dim, dtype=np.float32)

    # Unigrams: each word activates a dimension with +/- sign
    for word in words:
        idx = _word_hash_index(word, dim)
        sign = _word_hash_sign(word)
        vec[idx] += sign

    # Bigrams: capture word order / adjacency
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i + 1]}"
        idx = _bigram_hash_index(bigram, dim)
        vec[idx] += _word_hash_sign(bigram) * 0.5

    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec.astype(np.float32)


# ── Combined encoding ──────────────────────────────────────────────

def encode_statement_combined(statement: str, dim: int = 32) -> np.ndarray:
    """Combine BOW (discriminative) + n-gram (semantic) encodings.

    First half: hashed bag-of-words (captures which exact words are present)
    Second half: n-gram encoding (captures word similarity / morphology)

    This gives the network both:
    - Exact word identity signals (BOW)
    - Fuzzy word similarity signals (n-gram)
    """
    half = dim // 2
    rest = dim - half
    bow = encode_statement_bow(statement, half)
    ngram = encode_statement_ngram(statement, rest)
    combined = np.concatenate([bow, ngram])
    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined = combined / norm
    return combined.astype(np.float32)


# ── Unified interface ──────────────────────────────────────────────

def encode_statement(statement: str, dim: int = 32, method: str = "combined") -> np.ndarray:
    """Encode a statement into a fixed-dimension vector.

    Args:
        method: "hash", "ngram", "bow", or "combined" (default)
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
    elif method == "ngram":
        return encode_statement_ngram(statement, dim)
    elif method == "bow":
        return encode_statement_bow(statement, dim)
    else:  # "combined"
        return encode_statement_combined(statement, dim)


def encode_batch(statements: Sequence[str], dim: int = 32, method: str = "combined") -> np.ndarray:
    """Encode a batch of statements. Returns shape (N, dim)."""
    return np.array([encode_statement(s, dim, method) for s in statements], dtype=np.float32)
