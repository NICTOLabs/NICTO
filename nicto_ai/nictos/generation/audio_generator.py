"""
Audio Generator with Neural Codec and Diffusion Vocoder.

Generates music, speech, and sound effects using diffusion-based synthesis.
Architecture:
  - Neural Codec: Discrete audio tokens (like SoundStream/EnCodec)
  - DiT Backbone: Diffusion over codec tokens
  - HiFi-GAN Vocoder: Waveform synthesis from codec tokens
  - Music Conditioning: Style, tempo, key, dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==============================================================================
# Neural Audio Codec (VQ-VAE style)
# ==============================================================================

class ResidualVectorQuantization(nn.Module):
    """Residual Vector Quantization for audio tokens."""

    def __init__(self, codebook_size: int = 1024, num_codebooks: int = 8, dim: int = 512):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_codebooks)
        ])

        self.input_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vector quantize input."""
        B, T, D = x.shape

        residual = x
        codes = []
        committed_loss = 0.0

        for i in range(self.num_codebooks):
            # Find nearest codebook entry
            dists = torch.cdist(residual, self.codebooks[i].weight.unsqueeze(0))
            idx = dists.argmin(dim=-1)

            # Get quantized output
            quantized = self.codebooks[i](idx)
            codes.append(idx)

            # Update residual
            residual = residual - quantized
            committed_loss = committed_loss + F.mse_loss(quantized, x.detach())

        # Stack codes
        codes = torch.stack(codes, dim=-1)  # (B, T, num_codebooks)

        return codes, committed_loss

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        """Dequantize codes back to continuous."""
        B, T, K = codes.shape
        x = torch.zeros(B, T, self.codebooks[0].embedding_dim, device=codes.device)

        for i in range(K):
            x = x + self.codebooks[i](codes[:, :, i])

        return x


class AudioEncoder(nn.Module):
    """Encode audio waveform to neural codec tokens."""

    def __init__(self, in_channels: int = 1, dim: int = 512, codebook_size: int = 1024, num_codebooks: int = 8):
        super().__init__()

        # 1D convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, stride=2, padding=3),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, 5, stride=2, padding=2),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Conv1d(512, dim, 3, stride=1, padding=1),
        )

        # RVQ
        self.rvq = ResidualVectorQuantization(codebook_size, num_codebooks, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to tokens.

        Args:
            x: (B, 1, T) audio waveform

        Returns:
            codes: (B, T', K) quantized codes
            committed_loss: Scalar RVQ commitment loss
        """
        h = self.encoder(x)  # (B, dim, T')
        h = h.transpose(1, 2)  # (B, T', dim)

        codes, committed_loss = self.rvq.quantize(h)

        return codes, committed_loss


class AudioDecoder(nn.Module):
    """Decode neural codec tokens to audio waveform."""

    def __init__(self, out_channels: int = 1, dim: int = 512):
        super().__init__()

        # 1D convolutional decoder (HiFi-GAN style)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim, 512, 4, stride=1, padding=0),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, out_channels, 7, padding=3),
            nn.Tanh(),
        )

        # Multi-period discriminator for HiFi-GAN
        self.discriminators = nn.ModuleList([
            nn.Conv1d(out_channels, 1, k, padding=k//2)
            for k in [2, 3, 5, 7, 11]
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Decode features to waveform.

        Args:
            h: (B, dim, T') feature tensor

        Returns:
            (B, 1, T) audio waveform
        """
        return self.decoder(h)


# ==============================================================================
# Audio DiT (Diffusion Transformer for Audio)
# ==============================================================================

class AudioDiTBlock(nn.Module):
    """DiT block for audio generation."""

    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C), context: (B, S, context_dim), t_emb: (B, C)"""
        # Add time embedding
        x = x + t_emb.unsqueeze(1)

        # Self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention
        h = self.norm2(x)
        context_proj = self.context_proj(context)
        h, _ = self.cross_attn(h, context_proj, context_proj)
        x = x + h

        # FFN
        x = x + self.ffn(x)

        return x


# ==============================================================================
# Music Generator
# ==============================================================================

class MusicGenerator(nn.Module):
    """Music generation with style, tempo, key conditioning.

    Usage:
        generator = MusicGenerator()

        # Generate music
        audio = generator.generate(
            text_tokens,
            duration=30.0,  # seconds
            sample_rate=48000,
        )
    """

    def __init__(
        self,
        dim: int = 512,
        context_dim: int = 1024,
        codebook_size: int = 1024,
        num_codebooks: int = 8,
        num_layers: int = 12,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim

        # Audio codec
        self.encoder = AudioEncoder(dim=dim, codebook_size=codebook_size, num_codebooks=num_codebooks)
        self.decoder = AudioDecoder(dim=dim)
        self.rvq = ResidualVectorQuantization(codebook_size, num_codebooks, dim)

        # Input projection
        self.input_proj = nn.Linear(dim, dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Audio DiT blocks
        self.blocks = nn.ModuleList([
            AudioDiTBlock(dim, context_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, codebook_size)

        # Music-specific conditioning
        self.tempo_embed = nn.Embedding(8, dim)  # 8 tempo categories
        self.key_embed = nn.Embedding(12, dim)   # 12 musical keys
        self.style_embed = nn.Embedding(16, dim)  # 16 music styles

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        duration: float = 30.0,
        sample_rate: int = 48000,
        tempo: Optional[int] = None,
        key: Optional[int] = None,
        style: Optional[int] = None,
        cfg_scale: float = 7.5,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Generate music from text.

        Args:
            context: (B, S, context_dim) Text conditioning
            duration: Duration in seconds
            sample_rate: Audio sample rate
            tempo: Tempo category (0-7)
            key: Musical key (0-11)
            style: Music style (0-15)
            cfg_scale: CFG scale
            num_steps: ODE solver steps

        Returns:
            (B, 1, T) audio waveform
        """
        B = context.shape[0]
        device = context.device

        # Calculate sequence length
        # 8x temporal compression from encoder, 48kHz sample rate
        seq_len = int(duration * sample_rate / 8 / 8)  # 8x compression from encoder

        # Add music conditioning
        if tempo is not None:
            context = context + self.tempo_embed(torch.tensor(tempo, device=device)).unsqueeze(1)
        if key is not None:
            context = context + self.key_embed(torch.tensor(key, device=device)).unsqueeze(1)
        if style is not None:
            context = context + self.style_embed(torch.tensor(style, device=device)).unsqueeze(1)

        # Start from noise
        z = torch.randn(B, seq_len, self.dim, device=device)

        # Euler ODE solver
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_emb = self.time_embed(t.view(-1, 1))

            # Apply DiT blocks
            h = z
            for block in self.blocks:
                h = block(h, context, t_emb)

            # Output projection
            h = self.output_norm(h)
            v = self.output_proj(h)

            # CFG
            if cfg_scale > 1.0:
                uncond_context = torch.zeros_like(context)
                h_uncond = z
                for block in self.blocks:
                    h_uncond = block(h_uncond, uncond_context, t_emb)
                h_uncond = self.output_norm(h_uncond)
                v_uncond = self.output_proj(h_uncond)
                v = v_uncond + cfg_scale * (v - v_uncond)

            z = z + v * dt

        # Dequantize to continuous
        codes = z.argmax(dim=-1)  # (B, seq_len)
        codes = codes.unsqueeze(-1).expand(-1, -1, 8)  # (B, seq_len, num_codebooks)
        h = self.rvq.dequantize(codes)

        # Decode to waveform
        h = h.transpose(1, 2)  # (B, dim, seq_len)
        audio = self.decoder(h)

        return audio


# ==============================================================================
# Sound Effects Generator
# ==============================================================================

class SFXGenerator(nn.Module):
    """Sound effects generation."""

    def __init__(self, dim: int = 512, context_dim: int = 1024, num_layers: int = 8):
        super().__init__()
        self.dim = dim

        # Simple audio decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            AudioDiTBlock(dim, context_dim, num_heads=8)
            for _ in range(num_layers)
        ])

        # Output
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, 1)  # Single channel audio

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        duration: float = 5.0,
        sample_rate: int = 48000,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Generate sound effects.

        Args:
            context: (B, S, context_dim) Text conditioning
            duration: Duration in seconds
            sample_rate: Audio sample rate
            num_steps: ODE solver steps

        Returns:
            (B, 1, T) audio waveform
        """
        B = context.shape[0]
        device = context.device

        # Calculate sequence length
        seq_len = int(duration * sample_rate / 8 / 8)

        # Start from noise
        z = torch.randn(B, seq_len, self.dim, device=device)

        # Euler ODE solver
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_emb = self.time_embed(t.view(-1, 1))

            # Apply DiT blocks
            h = z
            for block in self.blocks:
                h = block(h, context, t_emb)

            # Output projection
            h = self.output_norm(h)
            v = self.output_proj(h).squeeze(-1)

            z = z + v.unsqueeze(-1) * dt

        # Decode to waveform
        audio = self.output_proj(self.output_norm(z)).squeeze(-1)
        audio = audio.unsqueeze(1)  # (B, 1, T)

        return torch.tanh(audio)
