"""
NICTO-GAN Trainer
Training loop with progressive growing, mixed precision, and metrics.

Features:
- Progressive growing (4x4 -> target resolution)
- Mixed precision training (FP16/BF16)
- R3GAN loss with R1+R2 penalties
- FID tracking (optional)
- Checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import time
import json

from nicto_ai.gan.generator import NICTOGenerator
from nicto_ai.gan.discriminator import NICTODiscriminator
from nicto_ai.gan.loss import R3GANLoss
from nicto_ai.gan.config import GANConfig


class NICTOGANTrainer:
    """
    NICTO-GAN Training Loop.
    
    Usage:
        config = GANConfig(target_size=64)
        trainer = NICTOGANTrainer(config)
        trainer.train(dataset)
    """

    def __init__(self, config: GANConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.generator = NICTOGenerator(
            z_dim=config.z_dim,
            style_dim=config.style_dim,
            channels=config.gen_channels,
            target_size=config.target_size,
            use_attention=config.use_attention,
            use_moe=config.use_moe,
            moe_experts=config.moe_experts,
        ).to(self.device)

        self.discriminator = NICTODiscriminator(
            channels=config.disc_channels,
            input_size=config.target_size,
            use_attention=config.use_attention,
            consciousness=config.consciousness,
        ).to(self.device)

        # Loss
        self.criterion = R3GANLoss(
            r1_gamma=config.r1_gamma,
            r2_gamma=config.r2_gamma,
        )

        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.g_lr,
            betas=(0.5, 0.99),
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.d_lr,
            betas=(0.5, 0.99),
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        self.use_amp = self.device.type == "cuda"

        # State
        self.step = 0
        self.g_losses = []
        self.d_losses = []

        print(f"Generator: {self.generator.get_params_count():,} params")
        print(f"Discriminator: {self.discriminator.get_params_count():,} params")
        print(f"Device: {self.device}")

    def train(self, dataset, save_dir: str = "checkpoints/gan"):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        print(f"\nStarting training for {self.config.total_steps} steps...")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: G={self.config.g_lr}, D={self.config.d_lr}")
        print()

        start_time = time.time()

        while self.step < self.config.total_steps:
            for batch in dataloader:
                if self.step >= self.config.total_steps:
                    break

                if isinstance(batch, (list, tuple)):
                    real_images = batch[0].to(self.device)
                else:
                    real_images = batch.to(self.device)

                # Handle different image sizes
                if real_images.dim() == 3:
                    real_images = real_images.unsqueeze(0)

                # Train discriminator
                d_loss, r1, r2 = self._train_discriminator(real_images)

                # Train generator
                g_loss = self._train_generator(real_images.shape[0])

                # Log
                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item())

                if self.step % 10 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Step {self.step:5d} | "
                        f"G: {g_loss.item():.4f} | "
                        f"D: {d_loss.item():.4f} | "
                        f"R1: {r1.item():.4f} | "
                        f"R2: {r2.item():.4f} | "
                        f"{elapsed:.1f}s"
                    )
                    start_time = time.time()

                # Save checkpoint
                if self.step % self.config.save_every == 0 and self.step > 0:
                    self._save_checkpoint(save_path / f"step_{self.step}.pt")

                self.step += 1

        # Final save
        self._save_checkpoint(save_path / "final.pt")
        self._save_losses(save_path / "losses.json")
        print(f"\nTraining complete! Saved to {save_path}")

    def _train_discriminator(self, real_images: torch.Tensor) -> tuple:
        """Train discriminator for one step."""
        self.d_optimizer.zero_grad()

        with torch.amp.autocast("cuda") if self.use_amp else torch.no_grad().__enter__():
            # Generate fake images
            z = torch.randn(real_images.shape[0], self.config.z_dim, device=self.device)
            fake_images = self.generator(z).detach()

            # Get scores
            real_out = self.discriminator(real_images)
            fake_out = self.discriminator(fake_images)

            # Compute loss
            losses = self.criterion(
                real_scores=real_out["score"],
                fake_scores=fake_out["score"],
                real_images=real_images,
                fake_images=fake_images,
                generator=self.generator,
                discriminator=self.discriminator,
            )

        if self.scaler:
            self.scaler.scale(losses["d_loss"]).backward()
            self.scaler.unscale_(self.d_optimizer)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip)
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
        else:
            losses["d_loss"].backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip)
            self.d_optimizer.step()

        return losses["d_loss"], losses["r1"], losses["r2"]

    def _train_generator(self, batch_size: int) -> torch.Tensor:
        """Train generator for one step."""
        self.g_optimizer.zero_grad()

        z = torch.randn(batch_size, self.config.z_dim, device=self.device)

        with torch.amp.autocast("cuda") if self.use_amp else torch.no_grad().__enter__():
            fake_images = self.generator(z)
            fake_out = self.discriminator(fake_images)

            # Generator loss: fool discriminator
            g_loss = torch.nn.functional.softplus(-fake_out["score"]).mean()

        if self.scaler:
            self.scaler.scale(g_loss).backward()
            self.scaler.unscale_(self.g_optimizer)
            nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.grad_clip)
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
        else:
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.grad_clip)
            self.g_optimizer.step()

        return g_loss

    def _save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        torch.save({
            "step": self.step,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "config": self.config.__dict__,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def _save_losses(self, path: Path):
        """Save loss history."""
        with open(path, "w") as f:
            json.dump({
                "g_losses": self.g_losses,
                "d_losses": self.d_losses,
            }, f)

    def generate(self, num_samples=16, save_path=None):
        """Generate samples."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.z_dim, device=self.device)
            images = self.generator(z)
        self.generator.train()

        if save_path:
            # Save as grid
            grid = F.utils.make_grid(images, nrow=4, normalize=True, value_range=(-1, 1))
            from torchvision.utils import save_image
            save_image(grid, save_path)
            print(f"Saved samples to {save_path}")

        return images
