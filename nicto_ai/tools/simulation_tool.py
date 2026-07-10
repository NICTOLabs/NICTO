"""
NICTO AI - Simulation Tool
Run physics or abstract simulations via NICTOS engine.
"""

import logging
import json
from typing import Dict, Optional, List

from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class SimulationTool(Tool):
    """Run physics / system-dynamics simulations.

    Supports sub-commands:
      - ``simulate:hydrogen`` — hydrogen-atom Coulomb simulation
      - ``simulate:threebody`` — 3-body gravitational simulation
      - ``simulate:lv`` — Lotka-Volterra predator-prey
      - ``simulate:sir`` — SIR epidemiology model
      - Custom: ``simulate:custom|{"entities": [...], "rules": [...], "steps": 1000, "dt": 0.001}``
    """

    name = "simulator"
    description = "Run physics simulations (hydrogen atom, 3-body gravity, Lotka-Volterra, SIR epidemic) or custom simulations. Returns trajectory and energy."
    parameters = [
        ToolParameter(name="query", type="string", description="Simulation spec: 'hydrogen', 'threebody', 'lv', 'sir', 'custom|<json>'", required=True),
        ToolParameter(name="steps", type="integer", description="Number of simulation steps", required=False, default=5000),
        ToolParameter(name="dt", type="float", description="Time step", required=False),
    ]
    tags = ["simulation", "physics", "nictos", "reality"]
    timeout_seconds = 30.0

    def _execute(self, query: str, steps: int = 5000, dt: float = None) -> ToolResult:
        try:
            import numpy as np
            from nicto_ai.nictos.api.simulation import Nictos
            from nicto_ai.nictos.domains.physics.types import Particle
            from nicto_ai.nictos.domains.physics.rules import GravityRule, CoulombRule
            from nicto_ai.nictos.domains.system_dynamics.types import Stock
            from nicto_ai.nictos.domains.system_dynamics.rules import LotkaVolterraRule, SIRRule
            from nicto_ai.nictos.domains.physics.potentials import LennardJones, CoulombPotential
            from nicto_ai.nictos.knowledge import PHYSICAL_CONSTANTS

            kit = Nictos()
            q_lower = query.lower()

            # -- auto-detect simulation type from flexible text --
            if any(kw in q_lower for kw in ["hydrogen", "proton", "electron", "bohr"]):
                sim = kit.create_simulation("hydrogen", domain="physics")
                # SI units
                m_e = PHYSICAL_CONSTANTS["m_e"]
                m_p = PHYSICAL_CONSTANTS["m_p"]
                e_charge = PHYSICAL_CONSTANTS["e"]
                k_e = PHYSICAL_CONSTANTS["k_e"]
                a_0 = PHYSICAL_CONSTANTS["a_0"]
                v_orbit = e_charge ** 2 / (4 * np.pi * 8.854e-12 * 1.054e-34)  # ~2.19e6
                proton = Particle(id=0, mass=m_p, charge=e_charge, position=[0.0, 0.0, 0.0])
                electron = Particle(id=1, mass=m_e, charge=-e_charge, position=[a_0, 0.0, 0.0], velocity=[0.0, v_orbit, 0.0])
                sim.add_particles([proton, electron])
                sim.add_rule(CoulombRule(k_e=k_e))
                actual_dt = dt or 1e-18   # ~6 steps per Bohr orbit period
                history = sim.run(steps=min(steps, 2000), dt=actual_dt)
                return self._format_history(history, "Hydrogen atom (Coulomb)")

            elif any(kw in q_lower for kw in ["threebody", "three body", "three-body", "nbody", "n-body", "n body", "gravity"]):
                sim = kit.create_simulation("threebody", domain="physics")
                p1 = Particle(id=0, mass=1.0, position=[-1.0, 0.0, 0.0], velocity=[0.0, 0.5, 0.0])
                p2 = Particle(id=1, mass=1.0, position=[1.0, 0.0, 0.0], velocity=[0.0, -0.5, 0.0])
                p3 = Particle(id=2, mass=1.0, position=[0.0, 1.73, 0.0], velocity=[-0.5, 0.0, 0.0])
                sim.add_particles([p1, p2, p3])
                sim.set_potential("gravity", G=1.0)
                actual_dt = dt or 0.001
                history = sim.run(steps=steps, dt=actual_dt)
                return self._format_history(history, "3-body gravity")

            elif any(kw in q_lower for kw in ["lv", "lotka", "predator", "prey", "volterra"]):
                sim = kit.create_simulation("lotka_volterra", domain="system_dynamics")
                prey = Stock(id=0, name="Prey", initial_value=40.0)
                pred = Stock(id=1, name="Predator", initial_value=9.0)
                sim.add_entity(prey)
                sim.add_entity(pred)
                sim.add_rule(LotkaVolterraRule(prey_id=0, pred_id=1, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5))
                actual_dt = dt or 0.01
                history = sim.run(steps=steps, dt=actual_dt)
                return self._format_stock_history(history, "Lotka-Volterra predator-prey")

            elif any(kw in q_lower for kw in ["sir", "epidem", "infection", "susc", "recover"]):
                sim = kit.create_simulation("sir", domain="system_dynamics")
                S = Stock(id=0, name="Susceptible", initial_value=990.0)
                I = Stock(id=1, name="Infectious", initial_value=10.0)
                R = Stock(id=2, name="Recovered", initial_value=0.0)
                sim.add_entity(S)
                sim.add_entity(I)
                sim.add_entity(R)
                sim.add_rule(SIRRule(S_id=0, I_id=1, R_id=2, beta=0.3, gamma=0.1))
                actual_dt = dt or 0.1
                history = sim.run(steps=steps, dt=actual_dt)
                return self._format_stock_history(history, "SIR epidemiology")

            elif query.startswith("custom|"):
                spec_str = query[len("custom|"):]
                spec = json.loads(spec_str)
                name = spec.get("name", "custom")
                sim = kit.create_simulation(name, domain=spec.get("domain", "physics"))
                for e in spec.get("entities", []):
                    p = Particle(id=e["id"], mass=e.get("mass", 1.0), charge=e.get("charge", 0.0),
                                 position=e.get("position"), velocity=e.get("velocity"))
                    sim.add_particle(p)
                for r in spec.get("rules", []):
                    if r.get("type") == "gravity":
                        sim.add_rule(GravityRule(G=r.get("G", 6.674e-11)))
                    elif r.get("type") == "coulomb":
                        sim.add_rule(CoulombRule(k_e=r.get("k_e", 8.987e9)))
                actual_dt = dt or spec.get("dt", 0.001)
                actual_steps = spec.get("steps", steps)
                history = sim.run(steps=actual_steps, dt=actual_dt)
                return self._format_history(history, f"Custom simulation: {name}")

            else:
                return ToolResult(success=False, error=f"Unknown simulation: \"{query}\". Try: hydrogen, threebody, lv, sir")

        except Exception as e:
            logger.exception("Simulation failed")
            return ToolResult(success=False, error=str(e))

    def _format_history(self, history, title: str) -> ToolResult:
        if not history:
            return ToolResult(success=False, error="No simulation history generated")
        last = history[-1]
        output = {
            "simulation": title,
            "steps": len(history),
            "final_time": last.time,
            "energy": last.energy,
            "temperature": last.temperature,
            "entities": [],
        }
        for ent in last.entities:
            e_data = {"id": ent.id, "type": ent.type_name}
            if ent.position is not None:
                e_data["position"] = ent.position.tolist()
            if "velocity" in ent.attributes:
                e_data["velocity"] = ent.attributes["velocity"]
            if "mass" in ent.attributes:
                e_data["mass"] = ent.attributes["mass"]
            if "charge" in ent.attributes:
                e_data["charge"] = ent.attributes["charge"]
            if "value" in ent.attributes:
                e_data["value"] = ent.attributes["value"]
            if "name" in ent.attributes:
                e_data["name"] = ent.attributes["name"]
            output["entities"].append(e_data)
        return ToolResult(success=True, output=output, metadata={"steps": len(history), "final_time": last.time, "energy": last.energy})

    def _format_stock_history(self, history, title: str) -> ToolResult:
        if not history:
            return ToolResult(success=False, error="No simulation history generated")
        last = history[-1]
        output = {
            "simulation": title,
            "steps": len(history),
            "final_time": last.time,
            "stocks": [],
        }
        for ent in last.entities:
            output["stocks"].append({
                "id": ent.id,
                "name": ent.attributes.get("name", f"stock_{ent.id}"),
                "value": ent.attributes.get("value", 0.0),
            })
        # Add trajectory summary
        output["trajectory"] = []
        for idx in range(len(history[0].entities)):
            stock_data = {"id": history[0].entities[idx].id, "name": history[0].entities[idx].attributes.get("name", f"stock_{idx}")}
            stock_data["values"] = [s.entities[idx].attributes.get("value", 0.0) for s in history[::max(1, len(history)//100)]]
            output["trajectory"].append(stock_data)
        return ToolResult(success=True, output=output, metadata={"steps": len(history), "final_time": last.time})
