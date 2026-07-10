"""Neural bridge — differentiable simulation, dataset generation, surrogate interfaces,
and DeepSeek V4-inspired architectural innovations.

Connects NICTOS to NICTO's PyTorch model via several approaches:

1. **Differentiable integrator** (:class:`TorchVerlet`) — backprop through physics.
2. **Simulation dataset** (:class:`SimulationDataset`) — generate (state → next-state)
   training pairs for a neural surrogate or for embedding into NICTO's weights.
3. **Surrogate trainer** (:func:`train_surrogate`) — train a small MLP to predict
   simulation trajectories without running the engine.

DeepSeek V4 Architectural Innovations:

4. **Engram Conditional Memory** (:class:`EngramModule`) — O(1) hash-based memory
   lookup that separates static knowledge from dynamic reasoning.
5. **Manifold-Constrained Hyper-Connections** (:class:`ManifoldConstrainedHyperConnections`) —
   doubly stochastic residual mixing via Sinkhorn-Knopp for stable deep training.
6. **DeepSeek Sparse Attention** (:class:`DeepSeekSparseAttention`) — hybrid local +
   global sparse attention for efficient long-context processing.
"""

from nicto_ai.nictos.neural.integrator import TorchVerlet
from nicto_ai.nictos.neural.dataset import SimulationDataset, trajectory_to_tensors
from nicto_ai.nictos.neural.engram import EngramModule, EngramTransformerBlock
from nicto_ai.nictos.neural.mhc import (
    ManifoldConstrainedHyperConnections,
    MHCBlock,
    MHCModel,
    sinkhorn_knopp,
)
from nicto_ai.nictos.neural.dattention import (
    DeepSeekSparseAttention,
    DSABlock,
    DSAModel,
    LightningIndexer,
)

__all__ = [
    "TorchVerlet",
    "SimulationDataset",
    "trajectory_to_tensors",
    # DeepSeek V4 innovations
    "EngramModule",
    "EngramTransformerBlock",
    "ManifoldConstrainedHyperConnections",
    "MHCBlock",
    "MHCModel",
    "sinkhorn_knopp",
    "DeepSeekSparseAttention",
    "DSABlock",
    "DSAModel",
    "LightningIndexer",
]
