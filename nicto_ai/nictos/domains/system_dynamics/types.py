from nicto_ai.nictos.core.types import GenericType


class Stock(GenericType):
    """A stock/compartment in a system-dynamics model.

    Rules modify ``attributes["value"]`` each step via ``Effect(attr="value", delta=...)``.
    """

    def __init__(self, id: int, name: str, initial_value: float = 0.0, metadata: dict | None = None):
        super().__init__(
            id=id,
            type_name="stock",
            position=None,
            attributes={"name": name, "value": initial_value},
            metadata=metadata or {},
        )
