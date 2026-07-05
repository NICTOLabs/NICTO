"""
NICTO-GAN Configuration
Configs for different GPU setups.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GANConfig:
    """NICTO-GAN configuration."""

    # Model
    z_dim: int = 512
    style_dim: int = 512
    gen_channels: int = 256
    disc_channels: int = 256
    target_size: int = 64
    use_attention: bool = True
    use_moe: bool = True
    moe_experts: int = 4
    consciousness: bool = True

    # Loss
    r1_gamma: float = 10.0
    r2_gamma: float = 10.0

    # Training
    total_steps: int = 50000
    batch_size: int = 64
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    grad_clip: float = 1.0
    save_every: int = 5000

    # Progressive growing (optional)
    progressive: bool = False
    progressive_stages: int = 3


def get_colab_t4_config() -> GANConfig:
    """Config for Colab Free T4 (16GB VRAM)."""
    return GANConfig(
        z_dim=256,
        style_dim=256,
        gen_channels=128,
        disc_channels=128,
        target_size=64,
        use_attention=True,
        use_moe=True,
        moe_experts=4,
        consciousness=True,
        r1_gamma=10.0,
        r2_gamma=10.0,
        total_steps=20000,
        batch_size=64,
        g_lr=2e-4,
        d_lr=2e-4,
        grad_clip=1.0,
        save_every=2000,
    )


def get_colab_pro_config() -> GANConfig:
    """Config for Colab Pro A100 (40GB VRAM)."""
    return GANConfig(
        z_dim=512,
        style_dim=512,
        gen_channels=256,
        disc_channels=256,
        target_size=128,
        use_attention=True,
        use_moe=True,
        moe_experts=8,
        consciousness=True,
        r1_gamma=10.0,
        r2_gamma=10.0,
        total_steps=50000,
        batch_size=128,
        g_lr=2e-4,
        d_lr=2e-4,
        grad_clip=1.0,
        save_every=5000,
    )


def get_k8s_a100_config() -> GANConfig:
    """Config for k8s cluster with 8x A100-80GB."""
    return GANConfig(
        z_dim=512,
        style_dim=512,
        gen_channels=512,
        disc_channels=512,
        target_size=256,
        use_attention=True,
        use_moe=True,
        moe_experts=16,
        consciousness=True,
        r1_gamma=10.0,
        r2_gamma=10.0,
        total_steps=200000,
        batch_size=256,
        g_lr=2e-4,
        d_lr=2e-4,
        grad_clip=1.0,
        save_every=10000,
    )
