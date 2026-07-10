from abc import ABC, abstractmethod
import numpy as np
from nicto_ai.nictos.core.types import GenericType, Effect


class GenericRule(ABC):
    """Abstract rule that inspects entities and produces Effects each step."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, entities: list[GenericType], dt: float) -> list[Effect]:
        pass

    def potential_energy(self, entities: list[GenericType]) -> float:
        """Override to contribute potential energy for energy conservation checks."""
        return 0.0
