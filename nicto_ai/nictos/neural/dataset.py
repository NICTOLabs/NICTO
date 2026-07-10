"""
Generate PyTorch Datasets from NICTOS simulation trajectories.

Approach #4 — the AI model internalizes physics as latent representations
by training on millions of (state → next-state) pairs.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional

from nicto_ai.nictos.core.engine import SimulationState


def trajectory_to_tensors(
    history: list[SimulationState],
    include_attr: Optional[list[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a simulation history to (state, next_state) tensor pairs.

    Each state is a flat vector:
      [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, mass, charge, ...attrs]

    Returns
    -------
    X : (T-1, N * D)  input states
    Y : (T-1, N * D)  target (next-step) states
    """
    if len(history) < 2:
        raise ValueError("Need at least 2 states in history")

    base_attr = include_attr or ["mass", "charge"]
    sequences = []
    for s in history:
        vec = []
        for ent in s.entities:
            pos = ent.position if ent.position is not None else np.zeros(3)
            vel = ent.attributes.get("velocity", np.zeros(3))
            vec.extend(pos.flatten().tolist())
            vec.extend(vel)
            for attr in base_attr:
                vec.append(ent.attributes.get(attr, 0.0))
        sequences.append(torch.tensor(vec, dtype=torch.float32))

    X = torch.stack(sequences[:-1])
    Y = torch.stack(sequences[1:])
    return X, Y


class SimulationDataset(Dataset):
    """PyTorch Dataset of (state → next-state) transitions.

    Can be fed directly into NICTO's training loop so the model learns
    physics from simulation data.

    Example
    -------
    >>> from nicto_ai.nictos import Nictos
    >>> from nicto_ai.nictos.domains.physics.types import Particle
    >>> from nicto_ai.nictos.domains.physics.rules import CoulombRule

    >>> kit = Nictos()
    >>> sim = kit.create_simulation("train", domain="physics")
    >>> # ... add particles and rules ...
    >>> history = sim.run(steps=1000, dt=0.001)
    >>> dataset = SimulationDataset(history)
    >>> X, Y = dataset[0]
    """

    def __init__(
        self,
        history: list[SimulationState],
        include_attr: Optional[list[str]] = None,
    ):
        self.X, self.Y = trajectory_to_tensors(history, include_attr)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def generate_training_data(
    simulation_fn,
    num_trajectories: int = 100,
    steps_per_traj: int = 500,
    dt: float = 0.001,
    noise_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate multi-trajectory training data for a neural surrogate.

    Parameters
    ----------
    simulation_fn : callable
        A function that returns a simulation with entities/rules configured
        (but not yet run), e.g. ``lambda: _make_hydrogen()``.
    num_trajectories : int
        How many runs to aggregate.
    steps_per_traj : int
        Steps per run.
    dt : float
        Timestep.
    noise_scale : float
        If > 0, add Gaussian noise to initial conditions to augment data.

    Returns
    -------
    all_X, all_Y : (total_steps-1, N*D) tensors
    """
    import copy

    all_X, all_Y = [], []
    for _ in range(num_trajectories):
        sim = simulation_fn()
        if noise_scale > 0:
            for ent in sim._engine.entities:
                if ent.position is not None:
                    ent.position += np.random.randn(3) * noise_scale
                if "velocity" in ent.attributes:
                    ent.attributes["velocity"] = (
                        np.asarray(ent.attributes["velocity"]) + np.random.randn(3) * noise_scale
                    ).tolist()
        history = sim._engine.run(steps=steps_per_traj, dt=dt)
        X, Y = trajectory_to_tensors(history)
        all_X.append(X)
        all_Y.append(Y)

    return torch.cat(all_X, dim=0), torch.cat(all_Y, dim=0)
