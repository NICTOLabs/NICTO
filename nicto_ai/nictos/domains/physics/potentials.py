import numpy as np
from nicto_ai.nictos.core.potentials import GenericPotential


class LennardJones(GenericPotential):
    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, cutoff: float = 2.5):
        super().__init__(name="lennard_jones")
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def compute(self, pos_a: np.ndarray, pos_b: np.ndarray, **params) -> tuple[float, np.ndarray]:
        epsilon = params.get("epsilon", self.epsilon)
        sigma = params.get("sigma", self.sigma)
        dr = np.asarray(pos_a, dtype=np.float64) - np.asarray(pos_b, dtype=np.float64)
        r2 = np.dot(dr, dr)
        if r2 < 1e-30:
            return 0.0, np.zeros(3)
        r = np.sqrt(r2)
        if r > self.cutoff * sigma:
            return 0.0, np.zeros(3)
        sr = sigma / r
        sr6 = sr ** 6
        sr12 = sr6 * sr6
        energy = 4.0 * epsilon * (sr12 - sr6)
        force_mag = 4.0 * epsilon * (12.0 * sr12 - 6.0 * sr6) / r2
        force = force_mag * dr
        return energy, force


class CoulombPotential(GenericPotential):
    def __init__(self, k_e: float = 8.9875517923e9):
        super().__init__(name="coulomb")
        self.k_e = k_e

    def compute(self, pos_a: np.ndarray, pos_b: np.ndarray, **params) -> tuple[float, np.ndarray]:
        charge_a = params.get("charge_a", 0.0)
        charge_b = params.get("charge_b", 0.0)
        dr = np.asarray(pos_a, dtype=np.float64) - np.asarray(pos_b, dtype=np.float64)
        r2 = np.dot(dr, dr)
        if r2 < 1e-30:
            return 0.0, np.zeros(3)
        r = np.sqrt(r2)
        energy = self.k_e * charge_a * charge_b / r
        force = self.k_e * charge_a * charge_b / r2 * dr / r
        return energy, force
