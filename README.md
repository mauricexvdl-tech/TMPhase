# TMPhase — Phase Cancellation Architecture

**Optimierung durch Destruktion — Optimization through Destruction**

A bio-inspired architecture where filtering emerges as a side-effect of signal interference rather than explicit computation. Instead of computing everything and then filtering, two opposing signal pathways interfere — irrelevant information cancels itself out, and only the truth survives.

## Core Idea

```
Classical:     Signal → Compute → Filter → Output
Phase Cancel:  Signal_E ──→ ┐
                            ├── Interference ──→ Residual = Output
               Signal_I ──→ ┘
```

The key insight: filtering doesn't have to be an algorithmic step. When two opposing signals propagate on the same medium, filtering emerges as a **side-effect** — the irrelevant cancels itself.

## Architecture

### Layer 1: TruthSpace (6 co-evolving NEAT networks)

| Network | Input | Output | Function |
|---------|-------|--------|----------|
| Embedder_E (excitatory) | 32 (encoded statement) | 16 (position) | Maps input to confirming position |
| Embedder_I (inhibitory) | 32 (encoded statement) | 16 (position) | Maps input to contradicting position |
| Interference | 32 (2×position) | 16 (residual) | Learns context-dependent signal mixing |
| Boundary | 16 (residual position) | 1 (truth score) | Decision surface for true/false |
| Dynamics_fwd | 32 (2×position) | 16 (conclusion) | Forward inference from premises |
| Dynamics_ctr | 32 (2×position) | 16 (conclusion) | Adversarial/contrary inference |

### Layer 2: Pragmatics (3 networks)

| Network | Input | Output | Function |
|---------|-------|--------|----------|
| Reachability | 16 | 80 (5×16) | Which positions are reachable? |
| Utility | 32 | 1 | How useful is this position? |
| Planner | 32 | 16 | What's the next step toward a goal? |

### Co-Evolution

All networks evolve together via NEAT (NeuroEvolution of Augmenting Topologies):

- **Embedder_E** is rewarded when true statements survive cancellation
- **Embedder_I** is rewarded when false statements are cancelled
- **Interference** learns when to cancel fully, partially, or amplify
- **Boundary** learns to read residual signals

This creates a natural adversarial arms race — each generation, the excitatory path gets better at signaling truth, while the inhibitory path gets better at cancelling falsehood.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Training

```bash
python -m tmphase --generations 50 --population 100
```

Options:
- `--generations N` — Number of evolutionary generations (default: 50)
- `--population N` — Population size per network (default: 100)
- `--input-dim N` — Input encoding dimension (default: 32)
- `--position-dim N` — TruthSpace position dimension (default: 16)
- `--no-learned-interference` — Use simple additive interference instead of learned

### Programmatic API

```python
from tmphase.evolution.coevolution import PhaseCancellationSystem, SystemConfig
from tmphase.evolution.fitness import OracleDataset

# Create oracle dataset
oracle = OracleDataset()
oracle.add("water is wet", True)
oracle.add("water is dry", False)
# ... add more items

# Configure and create system
config = SystemConfig(
    input_dim=32,
    position_dim=16,
    population_size=100,
    use_learned_interference=True,
)
system = PhaseCancellationSystem(config=config, oracle=oracle)

# Train
system.train(generations=50, verbose=True)

# Evaluate new statements
result = system.evaluate_statement("fire burns")
print(f"Truth: {result['truth_score']:.3f}, Signal: {result['signal_strength']:.3f}")
```

## Testing

```bash
pytest tests/ -v
```

## Biological Inspiration

- **Inhibitory neurons (GABAergic)**: ~20% of cortical neurons exist to *cancel* other signals
- **Alpha waves**: Active suppression of irrelevant cortex areas (attention through cancellation)
- **Efference copy**: The brain cancels its own predictions from sensory input — only surprise propagates
- **Synaptic pruning**: The brain improves by *deleting* half its connections

## Project Structure

```
tmphase/
├── neat/               # NEAT neuroevolution framework
│   ├── genome.py       # Genome encoding (nodes, connections, mutations)
│   ├── network.py      # Feed-forward network decoded from genome
│   ├── species.py      # Speciation for diversity preservation
│   └── population.py   # Population management and selection
├── networks/           # Phase Cancellation network components
│   ├── embedder.py     # Dual-pathway embedders (E and I)
│   ├── interference.py # Interference network (additive or learned)
│   ├── boundary.py     # Truth/false boundary in TruthSpace
│   ├── dynamics.py     # Forward and contrary inference
│   └── pragmatics.py   # Reachability, Utility, Planner
├── evolution/          # Co-evolutionary training
│   ├── fitness.py      # Fitness functions + oracle dataset
│   └── coevolution.py  # Full training pipeline
├── utils/              # Utilities
│   ├── encoding.py     # Symbol/statement encoding
│   └── activations.py  # Activation function registry
└── __main__.py         # CLI entry point
```
