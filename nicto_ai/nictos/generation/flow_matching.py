"""
Flow Matching DiT (Diffusion Transformer).

Replaces DDPM with flow matching for faster, better quality generation.
Flow matching uses ODE-based denoising instead of stochastic sampling,
enabling 10-50x faster generation with higher quality.

Architecture:
  - DiT backbone: Transformer with self-attention + cross-attention
  - Flow matching: Learn velocity field v(x_t, t) instead of noise ε(x_t, t)
  - ODE solver: Use Euler method for fast sampling (10-20 steps vs 1000)
  - Cross-attention: Condition on all modalities (text, image, video, audio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==============================================================================
# Building Blocks
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNModulation(nn.Module):
    """Adaptive Layer Norm modulation (AdaLN-Zero)."""

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 6),  # scale, shift, gate for 2 norms
        )
        self.hidden_dim = hidden_dim

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.mlp(t_emb)
        scale1, shift1, gate1, scale2, shift2, gate2 = out.chunk(6, dim=-1)
        return scale1, shift1, gate1, scale2, shift2, gate2


class SelfAttention(nn.Module):
    """Multi-head self-attention with AdaLN."""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        h = self.norm1(x)

        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, C)

        return x + self.proj(attn)

    def _forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward without internal norm - returns just the attention output (no residual)."""
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, C)
        return self.proj(attn)


class CrossAttention(nn.Module):
    """Multi-head cross-attention for conditioning."""

    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(context_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.kv = nn.Linear(context_dim, hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        S = context.shape[1]

        h1 = self.norm1(x)
        h2 = self.norm2(context)

        q = self.q(h1).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(h2).reshape(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, C)

        return x + self.proj(attn)

    def _forward_raw(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward without internal norms (norms applied by caller)."""
        B, L, C = x.shape
        S = context.shape[1]
        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, C)
        return x + self.proj(attn)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation."""

    def __init__(self, hidden_dim: int, ff_dim: int = None):
        super().__init__()
        ff_dim = ff_dim or hidden_dim * 4
        self.norm = nn.LayerNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, ff_dim * 2)  # GEGLU
        self.w2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = F.gelu(h1) * h2
        return x + self.w2(h)

    def _forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward without internal norm (norm applied by caller)."""
        h1, h2 = self.w1(x).chunk(2, dim=-1)
        h = F.gelu(h1) * h2
        return x + self.w2(h)


# ==============================================================================
# DiT Block
# ==============================================================================

class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN, self-attn, cross-attn, FFN."""

    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.adaln = AdaLNModulation(hidden_dim, hidden_dim)
        self.self_attn = SelfAttention(hidden_dim, num_heads)
        self.cross_attn = CrossAttention(hidden_dim, context_dim, num_heads)
        self.ffn = FeedForward(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        gate1: torch.Tensor,
        scale2: torch.Tensor,
        shift2: torch.Tensor,
        gate2: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN-modulated self-attention (norm inside self_attn)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.self_attn._forward_raw(h)

        # Cross-attention (norms inside cross_attn)
        x = x + self.cross_attn(x, context)

        # FFN (norm inside ffn)
        x = x + self.ffn(x)

        return x


# ==============================================================================
# DiT (Diffusion Transformer)
# ==============================================================================

class DiT(nn.Module):
    """Diffusion Transformer for flow matching.

    Learns the velocity field v(x_t, t) for ODE-based sampling.

    Usage:
        dit = DiT(hidden_dim=768, num_layers=12)

        # Training
        x0 = vae.encode(images)  # Clean latents
        x1 = torch.randn_like(x0)  # Noise
        t = torch.rand(B)  # Random timestep
        x_t = (1 - t) * x0 + t * x1  # Interpolation
        v_target = x1 - x0  # Target velocity
        v_pred = dit(x_t, t, context=text_features)
        loss = F.mse_loss(v_pred, v_target)

        # Sampling (ODE solver)
        x = torch.randn(B, latent_dim, H, W)  # Start from noise
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps)
            v = dit(x, t, context=text_features)
            x = x + v / num_steps  # Euler step
        image = vae.decode(x)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        context_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 8,
        patch_size: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Input projection
        self.input_proj = nn.Conv2d(512, hidden_dim, 1)  # latent_dim -> hidden_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, context_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 512)  # hidden_dim -> latent_dim
        self.final_conv = nn.Conv2d(512, 512, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            x_t: (B, latent_dim, H, W) noisy latent at timestep t
            t: (B,) timestep in [0, 1]
            context: (B, S, context_dim) conditioning from all modalities

        Returns:
            (B, latent_dim, H, W) predicted velocity
        """
        B, C, H, W = x_t.shape

        # Input projection
        h = self.input_proj(x_t)  # (B, hidden_dim, H, W)
        h = h.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)

        # Time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)

        # DiT blocks
        for block in self.blocks:
            scale1, shift1, gate1, scale2, shift2, gate2 = block.adaln(t_emb)
            h = block(h, context, scale1, shift1, gate1, scale2, shift2, gate2)

        # Output projection
        h = self.output_norm(h)
        h = self.output_proj(h)  # (B, H*W, latent_dim)
        h = h.transpose(1, 2).reshape(B, -1, H, W)  # (B, latent_dim, H, W)
        h = self.final_conv(h)

        return h


# ==============================================================================
# Flow Matching Trainer
# ==============================================================================

class FlowMatchingTrainer(nn.Module):
    """Flow matching trainer for the DiT.

    Trains the DiT to predict the velocity field v(x_t, t).

    Usage:
        trainer = FlowMatchingTrainer(dit, vae)

        # Training step
        images = batch["images"]
        loss = trainer.compute_loss(images, modality="image")
        loss.backward()
        optimizer.step()
    """

    def __init__(self, dit: DiT, vae: UnifiedVAE, sigma_min: float = 0.0002):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.sigma_min = sigma_min

    def compute_loss(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        modality: str = "image",
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            x0: (B, ...) Clean data (will be encoded by VAE)
            context: (B, S, context_dim) Conditioning
            modality: Modality for VAE encoding

        Returns:
            Scalar loss
        """
        # Encode clean data to latent space
        with torch.no_grad():
            mean, logvar = self.vae.encode(x0, modality)
            z0 = self.vae.reparameterize(mean, logvar)

        B = z0.shape[0]

        # Sample random timestep
        t = torch.rand(B, device=z0.device)

        # Sample noise
        z1 = torch.randn_like(z0)

        # Interpolate (linear interpolation path)
        t_expand = t.view(-1, 1, 1, 1)
        z_t = (1 - t_expand) * z0 + t_expand * z1

        # Target velocity
        v_target = z1 - z0

        # Predict velocity
        v_pred = self.dit(z_t, t, context)

        # Flow matching loss (velocity matching)
        loss = F.mse_loss(v_pred, v_target)

        return loss


# ==============================================================================
# Flow Matching Sampler
# ==============================================================================

class FlowMatchingSampler:
    """Fast ODE sampler for flow matching.

    Uses Euler method for fast sampling (10-20 steps vs 1000 for DDPM).
    """

    def __init__(self, dit: DiT, vae: UnifiedVAE, num_steps: int = 20):
        self.dit = dit
        self.vae = vae
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        modality: str = "image",
        latent_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Generate samples via ODE solving.

        Args:
            context: (B, S, context_dim) Conditioning
            modality: Output modality
            latent_shape: Shape of latent (if None, auto-determine)

        Returns:
            Generated tensor in modality space
        """
        B = context.shape[0]
        device = context.device

        # Determine latent shape
        if latent_shape is None:
            if modality == "image":
                latent_shape = (B, 512, 8, 8)  # 64x64 image -> 8x8 latent
            elif modality == "video":
                latent_shape = (B, 512, 2, 8, 8)  # 8s video -> 2 frames latent
            elif modality == "audio":
                latent_shape = (B, 512, 1, 16)  # 4s audio -> 16 samples latent

        # Start from noise
        z = torch.randn(latent_shape, device=device)

        # Euler ODE solver
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.dit(z, t, context)
            z = z + v * dt

        # Decode from latent space
        return self.vae.decode(z, modality)

    @torch.no_grad()
    def sample_with_cfg(
        self,
        context: torch.Tensor,
        uncond_context: torch.Tensor,
        modality: str = "image",
        cfg_scale: float = 7.5,
        latent_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Generate samples with classifier-free guidance.

        Args:
            context: (B, S, context_dim) Conditioning
            uncond_context: (B, S, context_dim) Unconditioned
            modality: Output modality
            cfg_scale: CFG scale (higher = more adherence to prompt)
            latent_shape: Shape of latent

        Returns:
            Generated tensor with CFG
        """
        B = context.shape[0]
        device = context.device

        # Determine latent shape
        if latent_shape is None:
            if modality == "image":
                latent_shape = (B, 512, 8, 8)
            elif modality == "video":
                latent_shape = (B, 512, 2, 8, 8)
            elif modality == "audio":
                latent_shape = (B, 512, 1, 16)

        # Start from noise
        z = torch.randn(latent_shape, device=device)

        # Euler ODE solver with CFG
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t = torch.full((B,), i * dt, device=device)

            # Conditional and unconditional predictions
            v_cond = self.dit(z, t, context)
            v_uncond = self.dit(z, t, uncond_context)

            # CFG
            v = v_uncond + cfg_scale * (v_cond - v_uncond)

            z = z + v * dt

        # Decode from latent space
        return self.vae.decode(z, modality)
