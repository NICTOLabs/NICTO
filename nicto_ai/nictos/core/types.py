import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenericType:
    """A generic simulated entity. Position is optional so the same engine
    can simulate particles (position + velocity) or abstract stocks (no position)."""

    id: int
    type_name: str
    position: np.ndarray | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.position is not None:
            self.position = np.asarray(self.position, dtype=np.float64)

    def to_dict(self) -> dict:
        d = {"id": self.id, "type_name": self.type_name, "attributes": dict(self.attributes), "metadata": dict(self.metadata)}
        if self.position is not None:
            d["position"] = self.position.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "GenericType":
        pos = np.array(data.get("position")) if data.get("position") is not None else None
        return cls(id=data["id"], type_name=data["type_name"], position=pos, attributes=data.get("attributes", {}), metadata=data.get("metadata", {}))


@dataclass
class Effect:
    """Result of a rule applied to one entity.

    Two channels (can be used together or independently):
      - force  : physical force (N) for entities with mass/velocity
      - attr/delta : additive change to entity.attributes[attr] for
                     any domain (populations, economics, epidemiology...)
    """
    entity_id: int
    force: np.ndarray | None = None
    attr: str | None = None       # attribute name to modify
    delta: float | None = None    # additive change to attribute
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def value_change(self) -> float | None:
        return self.delta

    @value_change.setter
    def value_change(self, v: float | None):
        self.delta = v
