import numpy as np
from nicto_ai.nictos.core.rules import GenericRule
from nicto_ai.nictos.core.types import GenericType, Effect


class GravityRule(GenericRule):
    def __init__(self, G: float = 6.674e-11):
        super().__init__(name="gravity")
        self.G = G

    def apply(self, entities: list[GenericType], dt: float) -> list[Effect]:
        effects = []
        for i, a in enumerate(entities):
            force = np.zeros(3, dtype=np.float64)
            for j, b in enumerate(entities):
                if i == j:
                    continue
                if a.position is None or b.position is None:
                    continue
                if "mass" not in a.attributes or "mass" not in b.attributes:
                    continue
                dr = np.asarray(b.position, dtype=np.float64) - np.asarray(a.position, dtype=np.float64)
                r2 = np.dot(dr, dr)
                if r2 < 1e-30:
                    continue
                r = np.sqrt(r2)
                # F = G m_a m_b / r^2  *  r_hat
                force += self.G * a.attributes["mass"] * b.attributes["mass"] / r2 * dr / r
            effects.append(Effect(entity_id=a.id, force=force))
        return effects

    def potential_energy(self, entities: list[GenericType]) -> float:
        pe = 0.0
        for i, a in enumerate(entities):
            for j, b in enumerate(entities):
                if i >= j:
                    continue
                if a.position is None or b.position is None:
                    continue
                if "mass" not in a.attributes or "mass" not in b.attributes:
                    continue
                dr = np.asarray(b.position, dtype=np.float64) - np.asarray(a.position, dtype=np.float64)
                r = np.sqrt(np.dot(dr, dr))
                if r < 1e-30:
                    continue
                pe -= self.G * a.attributes["mass"] * b.attributes["mass"] / r
        return pe


class CoulombRule(GenericRule):
    def __init__(self, k_e: float = 8.9875517923e9):
        super().__init__(name="coulomb")
        self.k_e = k_e

    def apply(self, entities: list[GenericType], dt: float) -> list[Effect]:
        effects = []
        for i, a in enumerate(entities):
            force = np.zeros(3, dtype=np.float64)
            for j, b in enumerate(entities):
                if i == j:
                    continue
                if a.position is None or b.position is None:
                    continue
                if "charge" not in a.attributes or "charge" not in b.attributes:
                    continue
                dr = np.asarray(b.position, dtype=np.float64) - np.asarray(a.position, dtype=np.float64)
                r2 = np.dot(dr, dr)
                if r2 < 1e-30:
                    continue
                r = np.sqrt(r2)
                # F = k_e q_a q_b / r^2  *  r_hat
                force += self.k_e * a.attributes["charge"] * b.attributes["charge"] / r2 * dr / r
            effects.append(Effect(entity_id=a.id, force=force))
        return effects

    def potential_energy(self, entities: list[GenericType]) -> float:
        pe = 0.0
        for i, a in enumerate(entities):
            for j, b in enumerate(entities):
                if i >= j:
                    continue
                if a.position is None or b.position is None:
                    continue
                if "charge" not in a.attributes or "charge" not in b.attributes:
                    continue
                dr = np.asarray(b.position, dtype=np.float64) - np.asarray(a.position, dtype=np.float64)
                r = np.sqrt(np.dot(dr, dr))
                if r < 1e-30:
                    continue
                pe += self.k_e * a.attributes["charge"] * b.attributes["charge"] / r
        return pe
