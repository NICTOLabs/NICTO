import numpy as np
from nicto_ai.nictos.core.types import GenericType


class Particle(GenericType):
    def __init__(self, id: int, mass: float, charge: float = 0.0, position=None, velocity=None, radius: float = 0.0, metadata: dict | None = None):
        attrs = {"mass": mass, "charge": charge, "radius": radius}
        if velocity is not None:
            attrs["velocity"] = list(np.asarray(velocity, dtype=np.float64))
        super().__init__(id=id, type_name="particle", position=position, attributes=attrs, metadata=metadata or {})


class Atom(GenericType):
    def __init__(self, id: int, element: str, mass: float, charge: float = 0.0, position=None, velocity=None, radius: float = 0.0, bonds: list | None = None, metadata: dict | None = None):
        attrs = {"element": element, "mass": mass, "charge": charge, "radius": radius, "atomic_number": _element_number(element)}
        if velocity is not None:
            attrs["velocity"] = list(np.asarray(velocity, dtype=np.float64))
        if bonds is not None:
            attrs["bonds"] = bonds
        super().__init__(id=id, type_name="atom", position=position, attributes=attrs, metadata=metadata or {})


def _element_number(element: str) -> int:
    table = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
             "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
             "K": 19, "Ca": 20, "Fe": 26, "Cu": 29, "Zn": 30, "Au": 79}
    return table.get(element, 0)
