"""
Train a lightweight neural network (surrogate) on NICTOS simulation data.

Approach #2 — once trained, the surrogate runs orders of magnitude faster
than the full engine while predicting physically plausible trajectories.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nicto_ai.nictos.neural.dataset import SimulationDataset


class MLPSurrogate(nn.Module):
    """Simple MLP that predicts the next state vector from current state.

    State vector (N particles × D features):
      [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, mass, charge, ...]
    """

    def __init__(self, state_dim: int, hidden: int = 256, layers: int = 4):
        super().__init__()
        seq = [nn.Linear(state_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            seq.append(nn.Linear(hidden, hidden))
            seq.append(nn.ReLU())
        seq.append(nn.Linear(hidden, state_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_surrogate(
    dataset: SimulationDataset,
    state_dim: int,
    hidden: int = 256,
    layers: int = 4,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> MLPSurrogate:
    """Train an MLPSurrogate on simulation data.

    Returns the trained model (state_dict can be saved and loaded into NICTO).
    """
    model = MLPSurrogate(state_dim, hidden, layers).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(f"  Surrogate epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.6e}")

    return model
