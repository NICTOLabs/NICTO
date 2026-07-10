import numpy as np
from nicto_ai.nictos.core.engine import SimulationEngine
from nicto_ai.nictos.core.integrators import verlet_step, symplectic_euler_step
from nicto_ai.nictos.core.io import save_state, load_state
from nicto_ai.nictos.core.types import GenericType
from nicto_ai.nictos.core.rules import GenericRule
from nicto_ai.nictos.domains.physics.types import Particle, Atom
from nicto_ai.nictos.domains.physics.rules import GravityRule, CoulombRule
from nicto_ai.nictos.domains.physics.dynamics import NBodyDynamics, MolecularDynamics
from nicto_ai.nictos.domains.physics.potentials import LennardJones, CoulombPotential


class Simulation:
    """High-level simulation interface.

    Wraps :class:`SimulationEngine` and provides convenience methods
    for configuring domains, adding entities, and running.
    """

    def __init__(self, domain: str = "physics", backend: str = "cpu"):
        self.domain = domain
        self.backend = backend
        self._engine = SimulationEngine()
        self._integrator = verlet_step

    # ---------- entity registration ----------

    def add_entity(self, entity: GenericType):
        self._engine.add_entity(entity)

    def add_particle(self, particle: Particle):
        self._engine.add_entity(particle)

    def add_atom(self, atom: Atom):
        self._engine.add_entity(atom)

    def add_particles(self, particles: list[Particle]):
        for p in particles:
            self._engine.add_entity(p)

    def add_rule(self, rule: GenericRule):
        self._engine.add_rule(rule)

    # ---------- potential shortcuts ----------

    def set_potential(self, potential_type: str, **params):
        if potential_type == "lennard_jones":
            self._engine.add_rule(NBodyDynamics(G=params.get("G", 6.674e-11)).create_simulation([]).rules[0])
        elif potential_type == "coulomb":
            self._engine.add_rule(CoulombRule(k_e=params.get("k_e", 8.987e9)))
        elif potential_type == "gravity":
            self._engine.add_rule(GravityRule(G=params.get("G", 6.674e-11)))

    def set_integrator(self, integrator_type: str = "verlet"):
        if integrator_type == "verlet":
            self._integrator = verlet_step
        elif integrator_type == "euler":
            self._integrator = symplectic_euler_step
        else:
            self._integrator = None  # engine will use default

    # ---------- running ----------

    def run(self, steps: int = 1000, dt: float = 0.001, record_every: int = 1):
        self._engine.run(steps=steps, dt=dt, integrator=self._integrator, record_every=record_every)
        return self._engine.history

    # ---------- I/O ----------

    def save(self, path: str):
        save_state(self._engine.get_state(), path)

    def load(self, path: str):
        state = load_state(path)
        self._engine.entities = state.entities
        self._engine.time = state.time
        self._engine.step_count = state.step


class Nictos:
    """Top-level factory.  Each call to *create_simulation* returns a new
    :class:`Simulation` that can be configured independently."""

    def __init__(self):
        self.simulations: dict[str, Simulation] = {}

    def create_simulation(self, name: str, domain: str = "physics", backend: str = "cpu") -> Simulation:
        sim = Simulation(domain=domain, backend=backend)
        self.simulations[name] = sim
        return sim

    def get_simulation(self, name: str) -> Simulation | None:
        return self.simulations.get(name)
