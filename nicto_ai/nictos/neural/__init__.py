"""Neural bridge — differentiable simulation, dataset generation, surrogate interfaces.

Connects NICTOS to NICTO's PyTorch model via several of the approaches:

1. **Differentiable integrator** (:class:`TorchVerlet`) — backprop through physics.
2. **Simulation dataset** (:class:`SimulationDataset`) — generate (state → next-state)
   training pairs for a neural surrogate or for embedding into NICTO's weights.
3. **Surrogate trainer** (:func:`train_surrogate`) — train a small MLP to predict
   simulation trajectories without running the engine.
"""

from nicto_ai.nictos.neural.integrator import TorchVerlet
from nicto_ai.nictos.neural.dataset import SimulationDataset, trajectory_to_tensors

__all__ = [
    "TorchVerlet",
    "SimulationDataset",
    "trajectory_to_tensors",
]
