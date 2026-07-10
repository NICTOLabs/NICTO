# NICTOS Usage Guide

## Quick Start

### Hydrogen Atom Simulation

```python
from nicto_ai.nictos import Nictos
from nicto_ai.nictos.domains.physics.types import Particle
from nicto_ai.nictos.domains.physics.rules import CoulombRule
from nicto_ai.nictos.knowledge import PHYSICAL_CONSTANTS

kit = Nictos()
sim = kit.create_simulation("hydrogen", domain="physics")

proton = Particle(id=0, mass=1836.0, charge=1.0, position=[0.0, 0.0, 0.0])
electron = Particle(
    id=1, mass=1.0, charge=-1.0,
    position=[5.29e-11, 0.0, 0.0],
    velocity=[0.0, 2.19e6, 0.0]
)

sim.add_particles([proton, electron])
sim.add_rule(CoulombRule(k_e=PHYSICAL_CONSTANTS["k_e"]))
history = sim.run(steps=2000, dt=1e-18)

last = history[-1]
print(f"Energy: {last.energy:.4e} J")
print(f"Electron pos: {last.entities[1].position}")
```

### Predator-Prey (Lotka-Volterra)

```python
from nicto_ai.nictos import Nictos
from nicto_ai.nictos.domains.system_dynamics.types import Stock
from nicto_ai.nictos.domains.system_dynamics.rules import LotkaVolterraRule

kit = Nictos()
sim = kit.create_simulation("lv", domain="system_dynamics")

prey = Stock(id=0, name="Prey", initial_value=40.0)
predator = Stock(id=1, name="Predator", initial_value=9.0)

sim.add_entity(prey)
sim.add_entity(predator)
sim.add_rule(LotkaVolterraRule(prey_id=0, pred_id=1))

history = sim.run(steps=5000, dt=0.01)
for s in history[-1].entities:
    print(f"{s.attributes['name']}: {s.attributes['value']:.2f}")
```

### SIR Epidemic Model

```python
from nicto_ai.nictos import Nictos
from nicto_ai.nictos.domains.system_dynamics.types import Stock
from nicto_ai.nictos.domains.system_dynamics.rules import SIRRule

kit = Nictos()
sim = kit.create_simulation("sir", domain="system_dynamics")

sim.add_entity(Stock(id=0, name="Susceptible", initial_value=990.0))
sim.add_entity(Stock(id=1, name="Infectious", initial_value=10.0))
sim.add_entity(Stock(id=2, name="Recovered", initial_value=0.0))
sim.add_rule(SIRRule(S_id=0, I_id=1, R_id=2, beta=0.3, gamma=0.1))

history = sim.run(steps=2000, dt=0.1)
```

### 3-Body Gravity

```python
from nicto_ai.nictos import Nictos
from nicto_ai.nictos.domains.physics.types import Particle

kit = Nictos()
sim = kit.create_simulation("3body", domain="physics")

p1 = Particle(id=0, mass=1.0, position=[-1.0, 0.0, 0.0], velocity=[0.0, 0.5, 0.0])
p2 = Particle(id=1, mass=1.0, position=[1.0, 0.0, 0.0], velocity=[0.0, -0.5, 0.0])
p3 = Particle(id=2, mass=1.0, position=[0.0, 1.73, 0.0], velocity=[-0.5, 0.0, 0.0])

sim.add_particles([p1, p2, p3])
sim.set_potential("gravity", G=1.0)
history = sim.run(steps=10000, dt=0.001)
```

## CLI Usage

```bash
# Interactive chat with tool support
nicto chat

# Run a specific tool
nicto tool simulator hydrogen
nicto tool simulator lv
nicto tool web_search "quantum computing"

# Run a preset workflow
nicto workflow search_and_summarize "latest AI research"

# Knowledge base operations
nicto knowledge crawl https://en.wikipedia.org/wiki/Physics
nicto knowledge query "what is quantum entanglement"
nicto knowledge stats

# System info
nicto info
```

## Neural Bridge (Training Surrogates)

```python
import torch
from nicto_ai.nictos import Nictos
from nicto_ai.nictos.domains.physics.types import Particle
from nicto_ai.nictos.domains.physics.rules import CoulombRule
from nicto_ai.nictos.neural.dataset import SimulationDataset
from nicto_ai.nictos.neural.surrogate import train_surrogate

# 1. Generate training data
kit = Nictos()
sim = kit.create_simulation("train", domain="physics")
sim.add_particles([
    Particle(id=0, mass=1836.0, charge=1.0, position=[0, 0, 0]),
    Particle(id=1, mass=1.0, charge=-1.0, position=[5.29e-11, 0, 0],
             velocity=[0, 2.19e6, 0])
])
sim.add_rule(CoulombRule())
history = sim.run(steps=2000, dt=1e-18)

# 2. Create dataset
dataset = SimulationDataset(history)
X, Y = dataset[0]  # (state_dim,) tensors

# 3. Train surrogate
model = train_surrogate(dataset, state_dim=X.shape[0], epochs=100)

# 4. Use trained model
with torch.no_grad():
    pred = model(X.unsqueeze(0).float())
```

## Differentiable Simulation

```python
import torch
from nicto_ai.nictos.neural.integrator import TorchVerlet, force_coulomb

verlet = TorchVerlet()

x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
v = torch.zeros(2, 3)
mass = torch.ones(2, 1)
charges = torch.tensor([[1.0], [-1.0]])

# Forward pass with gradient tracking
for _ in range(100):
    f = force_coulomb(x, charges)
    x, v, _ = verlet(x, v, f, mass, dt=0.001)

# Backprop through the simulation
loss = x.sum()
loss.backward()
print(x.grad)  # gradients w.r.t. initial positions
```

## Custom Simulations (JSON Spec)

```json
{
  "name": "my_simulation",
  "domain": "physics",
  "entities": [
    {"id": 0, "mass": 1.0, "charge": 1.0, "position": [0, 0, 0]},
    {"id": 1, "mass": 1.0, "charge": -1.0, "position": [1, 0, 0], "velocity": [0, 0.5, 0]}
  ],
  "rules": [
    {"type": "coulomb", "k_e": 8.987e9}
  ],
  "steps": 1000,
  "dt": 0.001
}
```

Via CLI:
```bash
nicto tool simulator 'custom|{"entities": [...], "rules": [...]}'
```

## Save/Load State

```python
sim.save("checkpoint.json")
sim.load("checkpoint.json")
```
