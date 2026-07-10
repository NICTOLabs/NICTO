"""
AI bridge — programmatic simulation builder.

Allows NICTO or any AI to construct a simulation from a simple declarative
spec without writing Python.
"""

from nicto_ai.nictos.core.types import GenericType, Effect
from nicto_ai.nictos.core.rules import GenericRule
from nicto_ai.nictos.api.simulation import Simulation, Nictos
import numpy as np


# ---------------------------------------------------------------------------
#  Declarative domain builder
# ---------------------------------------------------------------------------

class DomainBuilder:
    """Build a :class:`Simulation` from a plain dict spec.

    Example
    -------
    >>> builder = DomainBuilder()
    >>> sim = builder.build({
    ...     "name": "threebody",
    ...     "domain": "physics",
    ...     "entities": [
    ...         {"id": 0, "mass": 1.0, "charge": 0.0, "position": [0, 0, 0], "velocity": [0, 0.5, 0]},
    ...         {"id": 1, "mass": 1.0, "charge": 0.0, "position": [1, 0, 0], "velocity": [0, -0.5, 0]},
    ...     ],
    ...     "rules": [
    ...         {"type": "gravity", "G": 1.0},
    ...     ],
    ...     "steps": 10000,
    ...     "dt": 0.001,
    ... })
    """

    def __init__(self):
        self.kit = Nictos()

    def build(self, spec: dict) -> Simulation:
        name = spec.get("name", "sim")
        domain = spec.get("domain", "physics")
        sim = self.kit.create_simulation(name, domain=domain)

        # create entities
        for e in spec.get("entities", []):
            if domain == "physics":
                from nicto_ai.nictos.domains.physics.types import Particle
                p = Particle(
                    id=e["id"],
                    mass=e.get("mass", 1.0),
                    charge=e.get("charge", 0.0),
                    position=e.get("position"),
                    velocity=e.get("velocity"),
                )
                sim.add_particle(p)
            else:
                ent = GenericType(id=e["id"], type_name=domain, position=e.get("position"))
                for k, v in e.items():
                    if k not in ("id", "position", "type_name"):
                        ent.attributes[k] = v
                sim.add_entity(ent)

        # create rules
        from nicto_ai.nictos.domains.physics.rules import GravityRule, CoulombRule
        for r in spec.get("rules", []):
            rtype = r["type"]
            if rtype == "gravity":
                sim.add_rule(GravityRule(G=r.get("G", 6.674e-11)))
            elif rtype == "coulomb":
                sim.add_rule(CoulombRule(k_e=r.get("k_e", 8.987e9)))

        return sim


# ---------------------------------------------------------------------------
#  Template library — pre-built specs for common scenarios
# ---------------------------------------------------------------------------

TEMPLATES = {
    "hydrogen": {
        "name": "hydrogen",
        "domain": "physics",
        "entities": [
            {"id": 0, "mass": 1836.0, "charge": 1.0, "position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
            {"id": 1, "mass": 1.0, "charge": -1.0, "position": [5.29e-11, 0.0, 0.0], "velocity": [0.0, 2.19e6, 0.0]},
        ],
        "rules": [{"type": "coulomb", "k_e": 8.987e9}],
        "steps": 5000,
        "dt": 1e-16,
    },
    "threebody": {
        "name": "threebody",
        "domain": "physics",
        "entities": [
            {"id": 0, "mass": 1.0, "charge": 0.0, "position": [-1.0, 0.0, 0.0], "velocity": [0.0, 0.4, 0.0]},
            {"id": 1, "mass": 1.0, "charge": 0.0, "position": [1.0, 0.0, 0.0], "velocity": [0.0, -0.4, 0.0]},
            {"id": 2, "mass": 1.0, "charge": 0.0, "position": [0.0, 1.73, 0.0], "velocity": [0.5, 0.0, 0.0]},
        ],
        "rules": [{"type": "gravity", "G": 1.0}],
        "steps": 20000,
        "dt": 0.001,
    },
}


def load_template(name: str) -> Simulation | None:
    """Load a pre-built template by name."""
    spec = TEMPLATES.get(name)
    if spec is None:
        return None
    return DomainBuilder().build(spec)
