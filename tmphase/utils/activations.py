"""Activation functions available for NEAT network nodes."""

from __future__ import annotations

import numpy as np

# Registry of activation functions
ACTIVATIONS: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        ACTIVATIONS[name] = fn
        return fn
    return decorator


@_register("sigmoid")
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


@_register("tanh")
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


@_register("relu")
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


@_register("identity")
def identity(x: np.ndarray) -> np.ndarray:
    return x


@_register("gauss")
def gauss(x: np.ndarray) -> np.ndarray:
    return np.exp(-x * x / 2.0)


@_register("sin")
def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


@_register("abs")
def abs_(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


@_register("step")
def step(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.0)


def get_activation(name: str) -> callable:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name!r}. Available: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]
