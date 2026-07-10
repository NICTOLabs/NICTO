from abc import ABC, abstractmethod
import numpy as np


class GenericPotential(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, pos_a: np.ndarray, pos_b: np.ndarray, **params) -> tuple[float, np.ndarray]:
        """Return (energy, force) between a and b."""
        pass
