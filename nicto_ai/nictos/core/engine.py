from copy import deepcopy
from nicto_ai.nictos.core.types import GenericType, Effect
from nicto_ai.nictos.core.rules import GenericRule
from nicto_ai.nictos.core.integrators import verlet_step
import numpy as np

BOLTZMANN = 1.380649e-23


class SimulationState:
    def __init__(self, time: float, step: int, entities: list[GenericType], energy: float | None = None, temperature: float | None = None):
        self.time = time
        self.step = step
        self.entities = deepcopy(entities)
        self.energy = energy
        self.temperature = temperature

    def to_dict(self) -> dict:
        return {"time": self.time, "step": self.step, "entities": [e.to_dict() for e in self.entities], "energy": self.energy, "temperature": self.temperature}


class SimulationEngine:
    """Domain-agnostic simulation engine.

    Drives a list of :class:`GenericRule` s each time-step, collects
    :class:`Effect` s, integrates positions (velocity Verlet) and applies
    attribute-value changes for non-physical domains.
    """

    def __init__(self):
        self.entities: list[GenericType] = []
        self.rules: list[GenericRule] = []
        self.time: float = 0.0
        self.step_count: int = 0
        self.history: list[SimulationState] = []
        self._entity_map: dict[int, GenericType] = {}

    # ---------- entity / rule management ----------

    def add_entity(self, entity: GenericType):
        self.entities.append(entity)
        self._entity_map[entity.id] = entity

    def add_rule(self, rule: GenericRule):
        self.rules.append(rule)

    def get_entity(self, entity_id: int) -> GenericType | None:
        return self._entity_map.get(entity_id)

    # ---------- stepping ----------

    def step(self, dt: float, integrator=None):
        """Advance one time-step.

        The default integrator (velocity Verlet via :func:`verlet_step`)
        uses stored accelerations and is symplectic — good energy conservation.

        Parameters
        ----------
        dt : float
            Time increment.
        integrator : callable or None
            Defaults to :func:`verlet_step`.
        """
        # ---- collect effects from all rules ----
        effects = []
        for rule in self.rules:
            effects.extend(rule.apply(self.entities, dt))

        # ---- apply value-change effects (non-physical) ----
        for e in effects:
            if e.attr is not None and e.delta is not None:
                ent = self._entity_map.get(e.entity_id)
                if ent is not None:
                    ent.attributes[e.attr] = ent.attributes.get(e.attr, 0.0) + e.delta

        # ---- integrate forces (physical) ----
        if effects and any(e.force is not None for e in effects):
            if integrator is None:
                integrator = verlet_step
            self.entities = integrator(self.entities, effects, dt)

        self.time += dt
        self.step_count += 1

    def run(self, steps: int, dt: float, integrator=None, record_every: int = 1):
        for i in range(steps):
            self.step(dt, integrator)
            if i % record_every == 0:
                self.history.append(self._capture_state())
        return self.history

    # ---------- diagnostics ----------

    def _capture_state(self) -> SimulationState:
        E = self._compute_energy()
        T = self._compute_temperature()
        return SimulationState(time=self.time, step=self.step_count, entities=self.entities, energy=E, temperature=T)

    def get_state(self) -> SimulationState:
        return self._capture_state()

    def _compute_energy(self) -> float | None:
        ke = 0.0
        for e in self.entities:
            if "velocity" in e.attributes and "mass" in e.attributes:
                v = np.asarray(e.attributes["velocity"], dtype=np.float64)
                ke += 0.5 * e.attributes["mass"] * np.dot(v, v)
        pe = 0.0
        for rule in self.rules:
            pe += rule.potential_energy(self.entities)
        return ke + pe

    def _compute_temperature(self) -> float | None:
        ke = 0.0
        n = 0
        for e in self.entities:
            if "velocity" in e.attributes and "mass" in e.attributes:
                v = np.asarray(e.attributes["velocity"], dtype=np.float64)
                ke += 0.5 * e.attributes["mass"] * np.dot(v, v)
                n += 1
        if n < 2:
            return None
        # equipartition: KE_per_particle = (3/2) k_B T  (3 dof)
        return (2.0 / 3.0) * ke / (n * BOLTZMANN)
