"""
Differentiable velocity-Verlet integrator in PyTorch.

Allows NICTO to backpropagate through simulation steps — approach #3
(differentiable simulation) and #2 (neural surrogate training).
"""

import torch
import torch.nn as nn


class TorchVerlet(nn.Module):
    """Differentiable velocity-Verlet step.

    Input shapes (batched or unbatched):
        x :  (N, 3) or (B, N, 3)  — positions
        v :  (N, 3) or (B, N, 3)  — velocities
        f :  (N, 3) or (B, N, 3)  — forces
        mass : (N, 1) or (B, N, 1) — masses

    Returns:
        x_new, v_new  with the same shapes.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        mass: torch.Tensor,
        dt: float | torch.Tensor = 0.001,
        stored_accel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x_new, v_new, a_new) where a_new is the acceleration at the
        new position (can be fed back as stored_accel on the next call for
        correct velocity Verlet).
        """
        a_new = f / mass.clamp(min=1e-30)
        a_old = stored_accel if stored_accel is not None else a_new

        x_new = x + v * dt + 0.5 * a_old * dt * dt
        v_new = v + 0.5 * (a_old + a_new) * dt

        return x_new, v_new, a_new.detach() if a_new.requires_grad else a_new


def force_coulomb(
    x: torch.Tensor,
    charges: torch.Tensor,
    k_e: float = 8.987e9,
) -> torch.Tensor:
    """Coulomb force on each particle from all others.

    x : (N, 3)    positions (in metres)
    charges : (N, 1)   charges (in Coulombs)

    Returns : (N, 3) force vectors.
    """
    N = x.shape[0]
    forces = torch.zeros_like(x)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dr = x[j] - x[i]
            r2 = (dr * dr).sum()
            if r2 < 1e-30:
                continue
            r = r2.sqrt()
            forces[i] += k_e * charges[i] * charges[j] / r2 * dr / r
    return forces


def force_gravity(
    x: torch.Tensor,
    masses: torch.Tensor,
    G: float = 6.674e-11,
) -> torch.Tensor:
    """Gravitational force on each particle."""
    N = x.shape[0]
    forces = torch.zeros_like(x)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dr = x[j] - x[i]
            r2 = (dr * dr).sum()
            if r2 < 1e-30:
                continue
            r = r2.sqrt()
            forces[i] += G * masses[i] * masses[j] / r2 * dr / r
    return forces
