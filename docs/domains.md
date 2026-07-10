# NICTOS Domains

NICTOS supports multiple simulation domains through its generic entity/rule system. Each domain provides specialized types and rules while sharing the same engine.

## Physics Domain

### Particle

```python
from nicto_ai.nictos.domains.physics.types import Particle

p = Particle(
    id=0,
    mass=1.0,           # kg
    charge=1.6e-19,     # Coulombs
    position=[0, 0, 0], # metres
    velocity=[1, 0, 0], # m/s
    radius=1e-10        # metres (optional, for visualization)
)
```

### Atom

```python
from nicto_ai.nictos.domains.physics.types import Atom

atom = Atom(
    id=0,
    element="H",        #元素符号
    mass=1.008,          # AMU
    charge=0.0,
    position=[0, 0, 0],
    bonds=[1]           # bonded to entity 1
)
```

Supported elements: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Fe, Cu, Zn, Au

### GravityRule

Newtonian gravity between all entity pairs:

```python
from nicto_ai.nictos.domains.physics.rules import GravityRule

rule = GravityRule(G=6.674e11)  # N m²/kg²
```

Force: F = G * m1 * m2 / r²

### CoulombRule

Electrostatic force between charged entities:

```python
from nicto_ai.nictos.domains.physics.rules import CoulombRule

rule = CoulombRule(k_e=8.987e9)  # N m²/C²
```

Force: F = k_e * q1 * q2 / r²

### Potentials

```python
from nicto_ai.nictos.domains.physics.potentials import LennardJones, CoulombPotential

lj = LennardJones(epsilon=1.0, sigma=1.0)  # Lennard-Jones
cp = CoulombPotential(k_e=8.987e9)          # Coulomb potential energy
```

### Dynamics Presets

```python
from nicto_ai.nictos.domains.physics.dynamics import NBodyDynamics, MolecularDynamics

nbody = NBodyDynamics(G=1.0)                    # N-body gravity
md = MolecularDynamics(epsilon=1.0, sigma=1.0)  # Molecular dynamics
```

## System Dynamics Domain

### Stock

Abstract quantity (no position). Used for populations, economics, epidemiology:

```python
from nicto_ai.nictos.domains.system_dynamics.types import Stock

population = Stock(
    id=0,
    name="Rabbits",
    initial_value=100.0
)
```

### LotkaVolterraRule

Predator-prey model:

```python
from nicto_ai.nictos.domains.system_dynamics.rules import LotkaVolterraRule

rule = LotkaVolterraRule(
    prey_id=0,      # entity ID of prey
    pred_id=1,      # entity ID of predator
    alpha=1.0,      # prey birth rate
    beta=0.1,       # predation rate
    delta=0.075,    # predator reproduction rate
    gamma=1.5       # predator death rate
)
```

Equations:
- dprey/dt = α·prey - β·prey·pred
- dpred/dt = δ·prey·pred - γ·pred

### SIRRule

SIR epidemic model:

```python
from nicto_ai.nictos.domains.system_dynamics.rules import SIRRule

rule = SIRRule(
    S_id=0,    # Susceptible entity
    I_id=1,    # Infectious entity
    R_id=2,    # Recovered entity
    beta=0.3,  # transmission rate
    gamma=0.1  # recovery rate
)
```

Equations:
- dS/dt = -β·S·I/N
- dI/dt = β·S·I/N - γ·I
- dR/dt = γ·I

### ExponentialGrowthRule

Simple exponential growth/decay:

```python
from nicto_ai.nictos.domains.system_dynamics.rules import ExponentialGrowthRule

rule = ExponentialGrowthRule(
    stock_id=0,
    rate=0.1   # positive = growth, negative = decay
)
```

Equation: dX/dt = r·X

## Custom Domains

You can create custom domains by subclassing `GenericType` and `GenericRule`:

```python
from nicto_ai.nictos.core.types import GenericType, Effect
from nicto_ai.nictos.core.rules import GenericRule

class Market(GenericType):
    def __init__(self, id, name, price, supply):
        super().__init__(
            id=id, type_name="market",
            attributes={"name": name, "price": price, "supply": supply}
        )

class SupplyDemandRule(GenericRule):
    def __init__(self, elasticity=0.5):
        super().__init__("supply_demand")
        self.elasticity = elasticity

    def apply(self, entities, dt):
        effects = []
        for ent in entities:
            if ent.type_name == "market":
                supply = ent.attributes["supply"]
                price = ent.attributes["price"]
                delta = -self.elasticity * (supply - 100) * dt
                effects.append(Effect(entity_id=ent.id, attr="price", delta=delta))
        return effects
```
