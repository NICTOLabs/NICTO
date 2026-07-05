"""
NICTO-GAN Loss
Based on R3GAN (NeurIPS 2024) regularized relativistic loss.

Key innovations:
- Relativistic discriminator: scores real vs fake in pairs
- R1 penalty: gradient penalty on real data
- R2 penalty: gradient penalty on fake data
- Softplus activation for smooth gradients
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Dict, Tuple


class R3GANLoss(nn.Module):
    """
    R3GAN Relativistic Loss with R1 + R2 Gradient Penalties.
    
    The relativistic discriminator outputs:
    - D(real): "how real is this real sample?" 
    - D(fake): "how real is this fake sample?"
    
    Generator wants D(fake) high (fool discriminator).
    Discriminator wants D(real) high and D(fake) low.
    
    R1 penalizes gradients on real data (stability).
    R2 penalizes gradients on fake data (prevents mode collapse).
    """

    def __init__(self, r1_gamma=10.0, r2_gamma=10.0):
        super().__init__()
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma

    def forward(
        self,
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor,
        real_images: torch.Tensor = None,
        fake_images: torch.Tensor = None,
        generator: nn.Module = None,
        discriminator: nn.Module = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator and discriminator losses.
        
        Args:
            real_scores: D(real) scores
            fake_scores: D(fake) scores  
            real_images: Real images (for R1 penalty)
            fake_images: Fake images (for R2 penalty)
            generator: G model (for R2 penalty)
            discriminator: D model (for penalties)
            
        Returns:
            dict with 'g_loss', 'd_loss', 'r1', 'r2'
        """
        # Relativistic loss with softplus
        # D wants: softplus(-D(real) + D(fake)) + softplus(D(fake) - D(real))
        # G wants: softplus(D(real) - D(fake))
        
        d_loss = (
            torch.nn.functional.softplus(-real_scores + fake_scores).mean() +
            torch.nn.functional.softplus(fake_scores - real_scores).mean()
        )
        
        g_loss = torch.nn.functional.softplus(real_scores - fake_scores).mean()

        result = {"g_loss": g_loss, "d_loss": d_loss, "r1": torch.tensor(0.0), "r2": torch.tensor(0.0)}

        # R1 penalty: gradient penalty on real data
        if self.r1_gamma > 0 and real_images is not None and discriminator is not None:
            real_images.requires_grad_(True)
            real_out = discriminator(real_images)["score"]
            r1 = self._gradient_penalty(real_out, real_images)
            result["r1"] = r1
            result["d_loss"] = result["d_loss"] + self.r1_gamma * r1

        # R2 penalty: gradient penalty on fake data
        if self.r2_gamma > 0 and fake_images is not None and generator is not None and discriminator is not None:
            fake_images = fake_images.detach().requires_grad_(True)
            fake_out = discriminator(fake_images)["score"]
            r2 = self._gradient_penalty(fake_out, fake_images)
            result["r2"] = r2
            result["d_loss"] = result["d_loss"] + self.r2_gamma * r2

        return result

    def _gradient_penalty(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty: ||grad(D(x))||^2."""
        gradients = autograd.grad(
            outputs=outputs.sum(),
            inputs=inputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        penalty = gradients.pow(2).mean()
        return penalty


class HingeLoss(nn.Module):
    """Alternative: Hinge loss for comparison."""

    def __init__(self):
        super().__init__()

    def forward(self, real_scores, fake_scores, **kwargs):
        d_loss = F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()
        g_loss = -fake_scores.mean()
        return {"g_loss": g_loss, "d_loss": d_loss, "r1": torch.tensor(0.0), "r2": torch.tensor(0.0)}
