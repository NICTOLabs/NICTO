"""
NICTO-GAN: Advanced Generative Adversarial Network
Built on R3GAN (NeurIPS 2024) with NICTO-specific innovations:
- MLA attention for global coherence
- MoE for diverse generation
- Consciousness-aware discriminator
- Progressive growing for scalability
"""

from nicto_ai.gan.generator import NICTOGenerator
from nicto_ai.gan.discriminator import NICTODiscriminator
from nicto_ai.gan.loss import R3GANLoss
from nicto_ai.gan.trainer import NICTOGANTrainer
from nicto_ai.gan.config import GANConfig

__all__ = [
    "NICTOGenerator",
    "NICTODiscriminator", 
    "R3GANLoss",
    "NICTOGANTrainer",
    "GANConfig",
]
