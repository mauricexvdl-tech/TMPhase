"""Benchmark datasets for rigorous testing.

Three difficulty levels:
1. EASY — Antonym pairs (current demo level)
2. MEDIUM — Diverse statements requiring world knowledge
3. HARD — Subtle, compositional, or tricky statements
"""

from __future__ import annotations

from ..evolution.fitness import OracleDataset


def create_easy_dataset() -> OracleDataset:
    """Antonym pairs — trivially separable by word patterns."""
    ds = OracleDataset()
    pairs = [
        ("water is wet", "water is dry"),
        ("the sun is hot", "the sun is cold"),
        ("fire burns", "fire freezes"),
        ("ice is cold", "ice is hot"),
        ("birds can fly", "rocks can fly"),
        ("fish live in water", "fish live in fire"),
        ("trees have leaves", "trees have wheels"),
        ("the earth orbits the sun", "the earth orbits the moon"),
        ("humans need oxygen", "humans need poison"),
        ("gravity pulls things down", "gravity pushes things up"),
    ]
    for true_stmt, false_stmt in pairs:
        ds.add(true_stmt, True)
        ds.add(false_stmt, False)
    return ds


def create_medium_dataset() -> OracleDataset:
    """Diverse factual statements — not simple antonym flips."""
    ds = OracleDataset()

    # True statements (varied domains)
    trues = [
        "water boils at one hundred degrees celsius",
        "the moon reflects sunlight",
        "diamonds are made of carbon",
        "sound travels faster in water than air",
        "the amazon river is in south america",
        "lightning is an electrical discharge",
        "plants produce oxygen during photosynthesis",
        "whales are mammals not fish",
        "iron is attracted by magnets",
        "the speed of light is constant in vacuum",
        "penguins live in the southern hemisphere",
        "earthquakes are caused by tectonic plates",
        "honey never spoils",
        "mercury is the closest planet to the sun",
        "glass is made from sand",
        "blood carries oxygen through the body",
        "wolves hunt in packs",
        "salt dissolves in water",
        "the heart pumps blood",
        "copper conducts electricity well",
    ]

    # False statements (not simple word swaps)
    falses = [
        "water boils at fifty degrees celsius",
        "the moon produces its own light",
        "diamonds are made of iron",
        "sound travels faster in air than water",
        "the amazon river is in europe",
        "lightning is a chemical reaction",
        "plants produce carbon dioxide during photosynthesis",
        "whales are fish not mammals",
        "wood is attracted by magnets",
        "the speed of light changes in vacuum",
        "penguins live in the arctic north pole",
        "earthquakes are caused by wind",
        "honey spoils within days",
        "venus is the closest planet to the sun",
        "glass is made from metal",
        "blood carries nitrogen through the body",
        "wolves are solitary hunters",
        "salt does not dissolve in water",
        "the liver pumps blood",
        "rubber conducts electricity well",
    ]

    for s in trues:
        ds.add(s, True)
    for s in falses:
        ds.add(s, False)
    return ds


def create_hard_dataset() -> OracleDataset:
    """Tricky statements — compositional, subtle, or counterintuitive."""
    ds = OracleDataset()

    trues = [
        "a tomato is a fruit not a vegetable",
        "hot water can freeze faster than cold water",
        "there are more trees on earth than stars in the milky way",
        "octopuses have three hearts",
        "bananas are berries but strawberries are not",
        "venus rotates in the opposite direction to most planets",
        "a day on venus is longer than a year on venus",
        "goldfish can distinguish different human faces",
        "the great wall of china is not visible from space",
        "scotland has more redheads per capita than ireland",
        "sharks are older than trees",
        "oxford university is older than the aztec empire",
        "cleopatra lived closer in time to smartphones than to the pyramids",
        "an astronaut grows taller in space",
        "dead skin cells contribute to household dust",
    ]

    falses = [
        "a tomato is a vegetable not a fruit",
        "cold water always freezes faster than hot water",
        "there are more stars in the milky way than trees on earth",
        "octopuses have one heart like humans",
        "strawberries are berries but bananas are not",
        "all planets rotate in the same direction",
        "a day on venus is shorter than a year on venus",
        "goldfish have a three second memory",
        "the great wall of china is visible from space",
        "ireland has more redheads per capita than any other country",
        "trees are older than sharks",
        "the aztec empire is older than oxford university",
        "cleopatra lived closer in time to the pyramids than to us",
        "gravity makes astronauts shorter in space",
        "household dust is mostly outdoor soil particles",
    ]

    for s in trues:
        ds.add(s, True)
    for s in falses:
        ds.add(s, False)
    return ds


def create_split_dataset(
    dataset: OracleDataset, train_ratio: float = 0.7, seed: int = 42
) -> tuple[OracleDataset, OracleDataset]:
    """Split a dataset into train and test sets, preserving true/false balance."""
    import random

    rng = random.Random(seed)
    true_items = list(dataset.true_items)
    false_items = list(dataset.false_items)
    rng.shuffle(true_items)
    rng.shuffle(false_items)

    n_true_train = max(1, int(len(true_items) * train_ratio))
    n_false_train = max(1, int(len(false_items) * train_ratio))

    train = OracleDataset()
    test = OracleDataset()

    for item in true_items[:n_true_train]:
        train.add(item.statement, True)
    for item in true_items[n_true_train:]:
        test.add(item.statement, True)
    for item in false_items[:n_false_train]:
        train.add(item.statement, False)
    for item in false_items[n_false_train:]:
        test.add(item.statement, False)

    return train, test
