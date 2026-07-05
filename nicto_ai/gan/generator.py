"""
NICTO-GAN Generator
Style-based generator with MLA attention and MoE layers.

Architecture:
- Mapping network: z -> w (latent to style)
- Style blocks: AdaIN + Conv + MLA attention + MoE
- Progressive growing support (4x4 -> target resolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ModulatedConv2d(nn.Module):
    """Modulated convolution (StyleGAN2-style) with weight demodulation."""

    def __init__(self, in_ch, out_ch, kernel_size, style_dim=512, demodulate=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.demodulate = demodulate
        self.style_proj = nn.Linear(style_dim, out_ch)

    def forward(self, x, w):
        b, c, h, w_dim = x.shape

        # Apply style modulation
        style = self.style_proj(w).view(b, self.out_ch, 1, 1)  # (b, out_ch, 1, 1)

        if self.demodulate:
            weight = self.conv.weight.unsqueeze(0)  # (1, out_ch, in_ch, k, k)
            style_expanded = style.view(b, self.out_ch, 1, 1, 1)
            weight = weight * style_expanded
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * demod
            # Use grouped conv for per-sample demodulation
            weight = weight.reshape(b * self.out_ch, self.in_ch, self.kernel_size, self.kernel_size)
            x = x.reshape(1, b * c, h, w_dim)
            out = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=b)
            return out.reshape(b, self.out_ch, h, w_dim)
        else:
            # Simple style scaling
            return self.conv(x) * style


class AdaIN(nn.Module):
    """Adaptive Instance Normalization."""

    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_bias = nn.Linear(style_dim, channels)

    def forward(self, x, w):
        h = self.norm(x)
        scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)
        bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + scale) + bias


class SimpleAttention(nn.Module):
    """Efficient self-attention for spatial features."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
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


class ExpertBlock(nn.Module):
    """Single expert: Conv + SwiGLU."""

    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, channels, 3, padding=1)
        self.gate = nn.Conv2d(channels, hidden_dim, 3, padding=1)

    def forward(self, x):
        return self.conv2(F.silu(self.conv1(x)) * self.gate(x))


class MoELayer(nn.Module):
    """Mixture of Experts with top-2 routing."""

    def __init__(self, channels, hidden_dim, n_experts=4, n_activated=2):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.experts = nn.ModuleList([ExpertBlock(channels, hidden_dim) for _ in range(n_experts)])
        self.gate = nn.Conv2d(channels, n_experts, 1)

    def forward(self, x):
        b, c, h, w_dim = x.shape
        gate_logits = self.gate(x)  # (b, n_experts, h, w)
        gate_logits = gate_logits.reshape(b, self.n_experts, h * w_dim)  # (b, n_experts, hw)

        # Top-k routing
        weights, indices = torch.topk(gate_logits, self.n_activated, dim=1)  # (b, k, hw)
        weights = F.softmax(weights, dim=1)

        # Compute expert outputs
        out = torch.zeros(b, c, h, w_dim, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)  # (b, c, h, w)
            for j in range(self.n_activated):
                pos_mask = (indices[:, j] == i).float()  # (b, hw)
                ew = pos_mask * weights[:, j]  # (b, hw)
                ew = ew.view(b, 1, h, w_dim)  # (b, 1, h, w)
                out = out + ew * expert_out

        return out


class StyleBlock(nn.Module):
    """Single style block: AdaIN + Conv + Attention + MoE."""

    def __init__(self, channels, style_dim, use_attention=True, use_moe=True,
                 moe_experts=4, moe_hidden=None):
        super().__init__()
        self.adain = AdaIN(style_dim, channels)
        self.conv = ModulatedConv2d(channels, channels, 3, style_dim=style_dim)
        self.noise = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU(0.2)

        self.attention = SimpleAttention(channels) if use_attention else None
        self.moe = MoELayer(channels, moe_hidden or channels * 4, moe_experts) if use_moe else None

    def forward(self, x, w):
        h = self.adain(x, w)
        h = self.conv(h, w)
        h = h + self.noise * torch.randn(h.shape[0], 1, h.shape[2], h.shape[3], device=h.device)
        h = self.act(h)

        if self.attention is not None:
            h = self.attention(h)
        if self.moe is not None:
            h = self.moe(h)

        return h


class MappingNetwork(nn.Module):
    """z -> w mapping network (8 MLP layers)."""

    def __init__(self, z_dim=512, style_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else style_dim
            layers.extend([
                nn.Linear(in_dim, style_dim),
                nn.LeakyReLU(0.2),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NICTOGenerator(nn.Module):
    """
    NICTO Generator: Style-based with MLA attention and MoE.
    
    Args:
        z_dim: Latent dimension
        style_dim: Style vector dimension  
        channels: Base channel count
        target_size: Target image size (must be power of 2)
        use_attention: Use self-attention in later blocks
        use_moe: Use MoE in later blocks
        moe_experts: Number of experts
        map_layers: Mapping network depth
    """

    def __init__(
        self,
        z_dim=512,
        style_dim=512,
        channels=256,
        target_size=64,
        use_attention=True,
        use_moe=True,
        moe_experts=4,
        map_layers=8,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.style_dim = style_dim
        self.target_size = target_size

        # Mapping network
        self.mapping = MappingNetwork(z_dim, style_dim, map_layers)

        # Initial constant (4x4)
        self.const = nn.Parameter(torch.randn(1, channels, 4, 4))

        # Style blocks (each doubles resolution)
        num_blocks = int(math.log2(target_size)) - 2  # 4 -> target
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            res = 4 * (2 ** i)
            use_att = use_attention and res >= 32
            use_m = use_moe and res >= 8
            self.blocks.append(StyleBlock(channels, style_dim, use_att, use_m, moe_experts))

        # To RGB
        self.to_rgb = ModulatedConv2d(channels, 3, 1, style_dim=style_dim)

    def forward(self, z, return_style=False):
        """
        Args:
            z: (b, z_dim) latent vector
            return_style: if True, also return w style vector
        Returns:
            image: (b, 3, H, W) in [-1, 1]
            w: (b, style_dim) if return_style
        """
        w = self.mapping(z)

        # Start from constant
        x = self.const.expand(z.shape[0], -1, -1, -1)

        # Apply style blocks
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x, w)

        # To RGB
        x = self.to_rgb(x, w)
        x = torch.tanh(x)

        if return_style:
            return x, w
        return x

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())
