"""
NICTO-GAN Discriminator
Spectral-normalized discriminator with self-attention and consciousness awareness.

Based on R3GAN (NeurIPS 2024) with:
- Spectral normalization for Lipschitz constraint
- Self-attention for global structure
- Modern ConvNeXt-style residual blocks
- Consciousness-aware uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict


class ConvBlock(nn.Module):
    """Conv + BN + LeakyReLU with optional spectral norm."""

    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, use_sn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)]
        if use_sn:
            layers[0] = nn.utils.spectral_norm(layers[0])
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block with spectral norm."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttention(nn.Module):
    """Self-attention layer."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.utils.spectral_norm(nn.Conv2d(channels, channels * 3, 1))
        self.proj = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return x + self.gamma * out


class NICTODiscriminator(nn.Module):
    """
    NICTO Discriminator: Spectral-norm + self-attention + consciousness.
    
    Args:
        channels: Base channel count
        input_size: Input image size
        use_attention: Use self-attention at 16x16 and above
        consciousness: Output uncertainty estimate
    """

    def __init__(
        self,
        channels=128,
        input_size=64,
        use_attention=True,
        consciousness=True,
    ):
        super().__init__()
        self.consciousness_enabled = consciousness

        # From RGB
        self.from_rgb = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, channels, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Down blocks
        self.block1 = ConvBlock(channels, channels * 2, 4, 2, 1)
        self.block2 = ConvBlock(channels * 2, channels * 4, 4, 2, 1)

        # Attention at 8x8
        self.attention = SelfAttention(channels * 4) if use_attention else None

        # Residual blocks
        self.res1 = ResBlock(channels * 4)
        self.res2 = ResBlock(channels * 4)

        # Final
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat_dim = channels * 4

        # Real/fake score
        self.score = nn.Linear(self.flat_dim, 1)

        # Consciousness head
        if consciousness:
            self.consciousness = nn.Sequential(
                nn.Linear(self.flat_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        else:
            self.consciousness = None

    def forward(self, x) -> Dict[str, torch.Tensor]:
        h = self.from_rgb(x)      # 64->32
        h = self.block1(h)        # 32->16
        h = self.block2(h)        # 16->8

        if self.attention is not None:
            h = self.attention(h)

        h = self.res1(h)
        h = self.res2(h)

        h = self.pool(h).flatten(1)  # (b, flat_dim)

        result = {"score": self.score(h)}

        if self.consciousness is not None:
            result["uncertainty"] = self.consciousness(h.detach())

        return result

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())
