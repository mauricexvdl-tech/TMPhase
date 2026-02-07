"""Benchmark datasets for rigorous testing.

Three difficulty levels + a large-scale dataset:
1. EASY — Antonym pairs (trivially separable)
2. MEDIUM — Diverse factual statements
3. HARD — Subtle, compositional, or counterintuitive
4. LARGE — 120 items for proper generalization testing
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


def create_large_dataset() -> OracleDataset:
    """120 items — large enough for proper generalization testing.

    Mixed domains: physics, biology, geography, chemistry, astronomy,
    history, everyday knowledge. No trivial antonym pairs.
    """
    ds = OracleDataset()

    trues = [
        # Physics
        "light bends when passing through glass",
        "sound cannot travel through a vacuum",
        "friction generates heat",
        "objects in motion tend to stay in motion",
        "black absorbs more heat than white",
        "electricity flows through copper wire easily",
        "ice floats on liquid water",
        "pressure increases with depth in water",
        # Biology
        "the human body has two hundred and six bones",
        "red blood cells carry oxygen",
        "mushrooms are fungi not plants",
        "dolphins are warm blooded animals",
        "spiders have eight legs",
        "bacteria are single celled organisms",
        "the brain uses about twenty percent of body energy",
        "dna contains genetic instructions",
        # Chemistry
        "water is made of hydrogen and oxygen",
        "iron rusts when exposed to moisture and air",
        "baking soda neutralizes acid",
        "carbon dioxide is heavier than oxygen",
        "gold does not rust or corrode",
        "mixing vinegar and baking soda produces gas",
        "nitrogen makes up most of the atmosphere",
        "helium is lighter than air",
        # Geography
        "the sahara is the largest hot desert",
        "australia is both a country and a continent",
        "the pacific ocean is the largest ocean",
        "mount everest is the tallest mountain above sea level",
        "the nile is one of the longest rivers in the world",
        "iceland uses geothermal energy for heating",
        "japan is an island nation in the pacific",
        "the dead sea is one of the saltiest bodies of water",
        # Astronomy
        "mars appears red because of iron oxide",
        "jupiter is the largest planet in our solar system",
        "the sun is a medium sized star",
        "a light year measures distance not time",
        "saturn has prominent rings made of ice and rock",
        "the milky way is a spiral galaxy",
        "pluto was reclassified as a dwarf planet",
        "comets have tails that point away from the sun",
        # Everyday
        "refrigerators keep food cold to slow bacteria growth",
        "soap works by breaking up grease and oils",
        "rubber tires provide traction on wet roads",
        "sunscreen protects skin from ultraviolet radiation",
        "yeast causes bread dough to rise",
        "thermometers measure temperature",
        "mirrors reflect light",
        "magnifying glasses concentrate sunlight",
        # History & general
        "the printing press was invented before the telephone",
        "ancient rome had a system of aqueducts for water",
        "the compass uses earths magnetic field for navigation",
        "penicillin was the first widely used antibiotic",
        "silk was originally produced only in china",
        "the wheel is one of the oldest inventions",
        "photography captures images using light",
        "telescopes make distant objects appear closer",
        # More biology
        "cats are obligate carnivores",
        "trees absorb carbon dioxide from the air",
        "the human heart beats about seventy times per minute",
    ]

    falses = [
        # Physics (wrong)
        "light travels in straight lines through all materials",
        "sound travels fastest in a vacuum",
        "friction reduces heat between surfaces",
        "objects in motion naturally slow down on their own",
        "white absorbs more heat than black",
        "glass is a good conductor of electricity",
        "ice sinks in liquid water",
        "pressure decreases with depth in water",
        # Biology (wrong)
        "the human body has one hundred bones",
        "white blood cells carry oxygen to tissues",
        "mushrooms are a type of plant",
        "dolphins are cold blooded like fish",
        "spiders have six legs like insects",
        "bacteria are complex multicellular organisms",
        "the stomach uses the most body energy",
        "proteins contain genetic instructions",
        # Chemistry (wrong)
        "water is made of carbon and hydrogen",
        "iron rusts when kept completely dry",
        "baking soda is a strong acid",
        "carbon dioxide is lighter than helium",
        "gold rusts quickly in moist air",
        "mixing vinegar and water produces gas",
        "oxygen makes up most of the atmosphere",
        "helium is heavier than air",
        # Geography (wrong)
        "antarctica is the largest hot desert",
        "australia is a country but not a continent",
        "the atlantic ocean is the largest ocean",
        "mount kilimanjaro is the tallest mountain above sea level",
        "the thames is one of the longest rivers in the world",
        "iceland relies entirely on fossil fuels for heating",
        "japan is a landlocked country in asia",
        "the dead sea has fresh water with no salt",
        # Astronomy (wrong)
        "mars appears blue because of water on its surface",
        "saturn is the largest planet in our solar system",
        "the sun is the largest star in the universe",
        "a light year measures time not distance",
        "jupiter has prominent rings made of ice and rock",
        "the milky way is an elliptical galaxy",
        "pluto is still classified as a major planet",
        "comet tails always point toward the sun",
        # Everyday (wrong)
        "refrigerators heat food to kill bacteria",
        "soap works by adding more grease to surfaces",
        "metal tires provide the best traction on wet roads",
        "sunscreen protects skin from infrared radiation only",
        "salt causes bread dough to rise",
        "thermometers measure air pressure",
        "mirrors absorb all light",
        "magnifying glasses scatter sunlight evenly",
        # History & general (wrong)
        "the telephone was invented before the printing press",
        "ancient rome had no system for transporting water",
        "the compass uses the moon for navigation",
        "aspirin was the first widely used antibiotic",
        "silk was originally produced only in egypt",
        "the wheel was invented in the twentieth century",
        "photography captures images using sound waves",
        "telescopes make distant objects appear smaller",
        # More biology (wrong)
        "cats are herbivores that eat only plants",
        "trees release carbon dioxide into the air at night",
        "the human heart beats about ten times per minute",
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
