# NICTOS Architecture

NICTOS (Neural Integrated Computational Theory Of Simulation) is the simulation engine powering NICTO AI. It provides a domain-agnostic framework for running physics, system-dynamics, and custom simulations.

## Core Concepts

### GenericType

The universal entity type. Every simulated object is a `GenericType`:

- **id** — unique integer identifier
- **type_name** — domain label (`"particle"`, `"atom"`, `"stock"`)
- **position** — 3D vector (optional — abstract domains like epidemiology skip it)
- **attributes** — key-value store (`mass`, `charge`, `velocity`, `value`, etc.)

### Effect

The result of applying a rule to an entity. Two independent channels:

| Channel | Fields | Use Case |
|---------|--------|----------|
| Physical | `force` (3D vector) | Particles, atoms, gravity, Coulomb |
| Value-change | `attr` + `delta` (scalar) | Populations, economics, epidemics |

Both channels can fire in the same step.

### GenericRule

Abstract base class. Each rule implements:

```python
def apply(self, entities: list[GenericType], dt: float) -> list[Effect]
def potential_energy(self, entities: list[GenericType]) -> float  # optional
```

### SimulationEngine

Drives the simulation loop:

1. Collects `Effect`s from all rules
2. Applies value-change effects (non-physical)
3. Integrates forces via velocity Verlet (physical)
4. Records state snapshots

## Integrators

| Integrator | Symplectic | Energy Conservation | Use |
|-----------|-----------|-------------------|-----|
| `verlet_step` | Yes | Excellent | Default for physics |
| `symplectic_euler_step` | Yes | Good | Fast approximation |

The velocity Verlet uses stored accelerations (previous-step) to avoid double-counting forces.

## Domains

### Physics (`nicto_ai.nictos.domains.physics`)

**Types:**
- `Particle(id, mass, charge, position, velocity, radius)`
- `Atom(id, element, mass, charge, position, velocity, bonds)`

**Rules:**
- `GravityRule(G)` — Newtonian gravity
- `CoulombRule(k_e)` — electrostatic force

**Potentials:**
- `LennardJones(epsilon, sigma)` — Lennard-Jones potential
- `CoulombPotential(k_e)` — Coulomb potential energy

**Dynamics presets:**
- `NBodyDynamics(G)` — N-body gravitational
- `MolecularDynamics(epsilon, sigma)` — molecular dynamics

### System Dynamics (`nicto_ai.nictos.domains.system_dynamics`)

**Types:**
- `Stock(id, name, initial_value)` — abstract quantity (no position)

**Rules:**
- `LotkaVolterraRule(prey_id, pred_id, alpha, beta, delta, gamma)` — predator-prey
- `SIRRule(S_id, I_id, R_id, beta, gamma)` — SIR epidemic model
- `ExponentialGrowthRule(stock_id, rate)` — exponential growth/decay

## Neural Bridge

The neural bridge connects NICTOS to PyTorch for differentiable simulation and surrogate training.

### TorchVerlet (`neural/integrator.py`)

Differentiable velocity Verlet. Accepts tensors, returns tensors, gradients flow through.

```python
from nicto_ai.nictos.neural.integrator import TorchVerlet, force_coulomb

verlet = TorchVerlet()
x, v, a = verlet(positions, velocities, forces, masses, dt=0.001, stored_accel=a_prev)
```

### SimulationDataset (`neural/dataset.py`)

Converts engine history to `(state, next_state)` tensor pairs for training:

```python
from nicto_ai.nictos.neural.dataset import SimulationDataset

dataset = SimulationDataset(history)  # list[SimulationState]
X, Y = dataset[0]  # tensors of shape (N*D,)
```

### MLPSurrogate (`neural/surrogate.py`)

Lightweight MLP that predicts next state from current state:

```python
from nicto_ai.nictos.neural.surrogate import MLPSurrogate, train_surrogate

model = train_surrogate(dataset, state_dim=16, epochs=100)
```

## AI Bridge (`nictos/ai/`)

Declarative simulation building. Describe a simulation in JSON, get a running engine:

```python
from nicto_ai.nictos.ai import DomainBuilder

builder = DomainBuilder()
sim = builder.from_spec({
    "domain": "physics",
    "entities": [...],
    "rules": [...]
})
history = sim.run(steps=1000, dt=0.001)
```

Template library includes `hydrogen` and `threebody` presets.

## Knowledge (`nictos/knowledge/`)

Built-in physical constants and element data:

```python
from nicto_ai.nictos.knowledge import PHYSICAL_CONSTANTS, ELEMENT_DATA

print(PHYSICAL_CONSTANTS["k_e"])  # 8.987e9
print(ELEMENT_DATA["H"]["mass"])  # 1.008
```

## API

The `Simulation` class wraps `SimulationEngine` with convenience methods:

```python
from nicto_ai.nictos import Nictos

kit = Nictos()
sim = kit.create_simulation("my_sim", domain="physics")
sim.add_particle(Particle(id=0, mass=1.0, charge=1.0, position=[0,0,0]))
sim.add_rule(CoulombRule())
history = sim.run(steps=1000, dt=0.001)
```

## File Layout

```
nicto_ai/nictos/
├── core/           # GenericType, GenericRule, SimulationEngine, integrators
├── domains/
│   ├── physics/    # Particle, Atom, GravityRule, CoulombRule, potentials
│   └── system_dynamics/  # Stock, LotkaVolterra, SIR, ExponentialGrowth
├── neural/         # TorchVerlet, SimulationDataset, MLPSurrogate
├── ai/             # DomainBuilder, templates
├── knowledge/      # PHYSICAL_CONSTANTS, ELEMENT_DATA
├── api/            # Simulation, Nictos (high-level interface)
└── visualization/  # Plotting utilities
```
