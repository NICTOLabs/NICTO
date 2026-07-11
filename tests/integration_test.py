"""Verify NICTOS is fully integrated into NICTO."""
import sys; sys.path.insert(0, ".")
import numpy as np

print("=" * 60)
print("NICTOS Integration Tests")
print("=" * 60)

# 1. Imports
from nicto_ai.nictos import Nictos
from nicto_ai.nictos.domains.physics.types import Particle
from nicto_ai.nictos.domains.physics.rules import CoulombRule, GravityRule
from nicto_ai.nictos.domains.system_dynamics.types import Stock
from nicto_ai.nictos.domains.system_dynamics.rules import LotkaVolterraRule, SIRRule
from nicto_ai.nictos.knowledge import PHYSICAL_CONSTANTS
print("1. Imports OK")

# 2. Hydrogen simulation with Coulomb (was buggy before fix)
kit = Nictos()
sim = kit.create_simulation("hydrogen", domain="physics")
proton = Particle(id=0, mass=1836.0, charge=1.0, position=[0.0, 0.0, 0.0])
electron = Particle(id=1, mass=1.0, charge=-1.0, position=[5.29e-11, 0.0, 0.0], velocity=[0.0, 2.19e6, 0.0])
sim.add_particles([proton, electron])
sim.set_potential("coulomb", k_e=PHYSICAL_CONSTANTS["k_e"])
history = sim.run(steps=500, dt=1e-17)
last = history[-1]
print("2. Hydrogen simulation OK")
print(f"   Steps: {len(history)}, Final time: {last.time:.4e}")
print(f"   Energy: {last.energy:.4e}")
print(f"   Electron pos: [{last.entities[1].position[0]:.4e}, {last.entities[1].position[1]:.4e}]")

# 3. SimulationTool
from nicto_ai.tools.simulation_tool import SimulationTool
tool = SimulationTool()
r = tool._execute("hydrogen", steps=500)
print(f"3. Tool hydrogen: success={r.success}")
if r.success:
    print(f"   Energy: {r.output.get('energy')}")

r = tool._execute("lv", steps=2000, dt=0.01)
print(f"4. Tool LV: success={r.success}")
if r.success:
    for s in r.output["stocks"]:
        print(f"   {s['name']}: {s['value']:.2f}")

r = tool._execute("sir", steps=2000, dt=0.1)
print(f"5. Tool SIR: success={r.success}")
if r.success:
    for s in r.output["stocks"]:
        print(f"   {s['name']}: {s['value']:.2f}")

# 6. ToolAgent intent + format
from nicto_ai.tools import create_default_registry
from nicto_ai.agent import ToolAgent
registry = create_default_registry()
agent = ToolAgent(registry)
resp = agent.process("simulate a hydrogen atom")
print(f"6. ToolAgent hydrogen: success={resp.success if resp else False}")
if resp:
    print(f"   tool={resp.tool_name}")
    print(f"   text preview: {resp.text[:200]}")

resp = agent.process("simulate predator prey")
print(f"7. ToolAgent LV: success={resp.success if resp else False}")
if resp:
    print(f"   text preview: {resp.text[:200]}")

# 8. Verify core engine directly
from nicto_ai.nictos.core.engine import SimulationEngine
from nicto_ai.nictos.core.types import GenericType, Effect
from nicto_ai.nictos.core.rules import GenericRule
from nicto_ai.nictos.core.integrators import verlet_step

class ConstForceRule(GenericRule):
    def __init__(self, force):
        super().__init__("const_force")
        self.force = np.asarray(force, dtype=np.float64)
    def apply(self, entities, dt):
        return [Effect(entity_id=e.id, force=self.force) for e in entities]

engine = SimulationEngine()
p = GenericType(id=0, type_name="particle", position=[0.0, 0.0, 0.0], attributes={"mass": 1.0, "velocity": [0.0, 0.0, 0.0]})
engine.add_entity(p)
engine.add_rule(ConstForceRule([1.0, 0.0, 0.0]))
history = engine.run(steps=10, dt=0.01)
assert abs(engine.time - 0.1) < 1e-12, f"time={engine.time}"
assert engine.step_count == 10, f"steps={engine.step_count}"
print(f"8. Core engine test: time={engine.time}, steps={engine.step_count}, pos={p.position}")
print("   PASS")

print()
print("=" * 60)
print("ALL INTEGRATION TESTS COMPLETE")
print("=" * 60)
