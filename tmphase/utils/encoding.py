"""Encoding utilities for converting symbolic inputs to numeric vectors."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


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


def encode_statement(statement: str, dim: int = 32) -> np.ndarray:
    """Encode a natural-language statement into a fixed-dimension vector.

    Combines word-level encodings via averaging then rescaling.
    """
    words = statement.lower().split()
    if not words:
        return np.zeros(dim, dtype=np.float32)
    vectors = [encode_symbol(w, dim) for w in words]
    combined = np.mean(vectors, axis=0)
    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined = combined / norm
    return combined.astype(np.float32)


def encode_batch(statements: Sequence[str], dim: int = 32) -> np.ndarray:
    """Encode a batch of statements. Returns shape (N, dim)."""
    return np.array([encode_statement(s, dim) for s in statements], dtype=np.float32)
