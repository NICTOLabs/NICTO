import numpy as np
from nicto_ai.nictos.core.engine import SimulationEngine
from nicto_ai.nictos.core.integrators import verlet_step
from nicto_ai.nictos.domains.physics.types import Particle, Atom
from nicto_ai.nictos.domains.physics.potentials import LennardJones, CoulombPotential
from nicto_ai.nictos.domains.physics.rules import GravityRule, CoulombRule


class NBodyDynamics:
    def __init__(self, G: float = 6.674e-11):
        self.G = G

    def create_simulation(self, particles: list[Particle]) -> SimulationEngine:
        engine = SimulationEngine()
        for p in particles:
            engine.add_entity(p)
        engine.add_rule(GravityRule(G=self.G))
        return engine

    def run(self, particles: list[Particle], steps: int, dt: float) -> SimulationEngine:
        engine = self.create_simulation(particles)
        engine.run(steps=steps, dt=dt, integrator=verlet_step)
        return engine


class MolecularDynamics:
    def __init__(self, epsilon: float = 0.650, sigma: float = 0.316, temperature: float = 300.0):
        self.lj = LennardJones(epsilon=epsilon, sigma=sigma)
        self.temperature = temperature

    def force_from_lj(self, atoms: list[Atom]) -> dict[int, np.ndarray]:
        forces: dict[int, np.ndarray] = {}
        for i, a in enumerate(atoms):
            f = np.zeros(3, dtype=np.float64)
            for j, b in enumerate(atoms):
                if i == j or a.position is None or b.position is None:
                    continue
                _, force = self.lj.compute(a.position, b.position)
                f += force
            forces[a.id] = f
        return forces

    def run(self, atoms: list[Atom], steps: int, dt: float) -> SimulationEngine:
        engine = SimulationEngine()
        for a in atoms:
            engine.add_entity(a)
        engine.add_rule(CoulombRule())
        engine.run(steps=steps, dt=dt, integrator=verlet_step)
        return engine
