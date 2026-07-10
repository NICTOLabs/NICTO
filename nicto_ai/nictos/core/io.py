import json
import numpy as np
from nicto_ai.nictos.core.types import GenericType
from nicto_ai.nictos.core.engine import SimulationState


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def save_state(state: SimulationState, path: str):
    with open(path, "w") as f:
        json.dump(state.to_dict(), f, cls=NumpyEncoder, indent=2)


def load_state(path: str) -> SimulationState:
    with open(path) as f:
        data = json.load(f)
    entities = [GenericType.from_dict(e) for e in data["entities"]]
    return SimulationState(time=data["time"], step=data["step"], entities=entities, energy=data.get("energy"), temperature=data.get("temperature"))
