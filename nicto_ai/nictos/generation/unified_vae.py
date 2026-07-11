"""
Unified VAE for all modalities.

Encodes text, image, video, and audio into a shared latent space.
Uses separate encoders per modality but shares the latent projection
and decoder across modalities for cross-modal consistency.

Architecture:
  - Image Encoder: Conv2d VAE (4x spatial compression)
  - Video Encoder: Conv3d VAE (4x spatial + 4x temporal compression)
  - Audio Encoder: 1D Conv VAE (8x temporal compression)
  - Text Encoder: BPE tokens projected to latent space
  - Shared Decoder: Conv2d decoder (transposed conv)
  - Unified Latent Space: all modalities map to same d-dimensional space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union


# ==============================================================================
# Building Blocks
# ==============================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, time_emb_dim: int = 0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, channels) if time_emb_dim > 0 else None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class ResBlock1D(nn.Module):
    """1D Residual block for audio processing."""

    def __init__(self, channels: int, time_emb_dim: int = 0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, channels) if time_emb_dim > 0 else None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(self.act(t_emb)).unsqueeze(-1)
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class ResBlock3D(nn.Module):
    """3D Residual block for video processing."""

    def __init__(self, channels: int, time_emb_dim: int = 0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, channels) if time_emb_dim > 0 else None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timestep conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# ==============================================================================
# Modality-Specific Encoders
# ==============================================================================

class ImageEncoder(nn.Module):
    """Encode images to latent vectors (4x spatial compression)."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 512, channels: list = None):
        super().__init__()
        channels = channels or [128, 256, 512, 1024]
        layers = []
        ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(),
                ResBlock(out_ch),
            ])
            ch = out_ch
        layers.append(nn.Conv2d(ch, latent_dim * 2, 3, padding=1))  # mean + logvar
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar) of the latent distribution."""
        params = self.net(x)
        mean, logvar = torch.chunk(params, 2, dim=1)
        return mean, logvar


class VideoEncoder(nn.Module):
    """Encode video to latent vectors (4x spatial + 4x temporal compression)."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 512, channels: list = None):
        super().__init__()
        channels = channels or [128, 256, 512, 1024]
        layers = []
        ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv3d(ch, out_ch, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(),
                ResBlock3D(out_ch),
            ])
            ch = out_ch
        # Temporal compression
        layers.extend([
            nn.Conv3d(ch, ch, (4, 1, 1), stride=(4, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
        ])
        layers.append(nn.Conv3d(ch, latent_dim * 2, 3, padding=1))  # mean + logvar
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar) of the latent distribution.

        Args:
            x: (B, C, T, H, W) video tensor
        """
        params = self.net(x)
        mean, logvar = torch.chunk(params, 2, dim=1)
        return mean, logvar


class AudioEncoder(nn.Module):
    """Encode audio to latent vectors (8x temporal compression)."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 512, channels: list = None):
        super().__init__()
        channels = channels or [128, 256, 512, 1024]
        layers = []
        ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(),
                ResBlock1D(out_ch),
            ])
            ch = out_ch
        layers.append(nn.Conv1d(ch, latent_dim * 2, 3, padding=1))  # mean + logvar
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar) of the latent distribution.

        Args:
            x: (B, 1, T) audio waveform
        """
        params = self.net(x)
        mean, logvar = torch.chunk(params, 2, dim=1)
        return mean, logvar


class TextEncoder(nn.Module):
    """Encode text tokens to latent vectors.

    Projects BPE tokens to the shared latent space.
    """

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 512, latent_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.proj = nn.Linear(embed_dim, latent_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode text tokens to latent vectors.

        Args:
            tokens: (B, seq_len) token IDs

        Returns:
            (B, seq_len, latent_dim) latent vectors
        """
        x = self.embed(tokens)
        h, _ = self.attn(x, x, x)
        h = self.norm(h + x)
        return self.proj(h)


class PointCloudEncoder(nn.Module):
    """Encode point clouds to latent vectors.

    Uses PointNet++ style set abstraction for hierarchical features.
    Input: (B, N, 6) - xyz (3) + rgb (3)
    Output: (mean, logvar) each (B, latent_dim)
    """

    def __init__(self, in_channels: int = 6, latent_dim: int = 512, channels: list = None):
        super().__init__()
        channels = channels or [64, 128, 256]
        layers = []
        ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Linear(ch, out_ch),
                nn.LayerNorm(out_ch),
                nn.SiLU(),
            ])
            ch = out_ch
        self.mlp = nn.ModuleList(layers)
        self.pool_proj = nn.Linear(ch, latent_dim * 2)  # mean + logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud to latent distribution.

        Args:
            x: (B, N, 6) point cloud (xyz + rgb)

        Returns:
            (mean, logvar) each (B, latent_dim)
        """
        h = x
        for layer in self.mlp:
            h = layer(h)
        # Global max + mean pooling
        h_max = h.max(dim=1)[0]
        h_mean = h.mean(dim=1)[0]
        h = h_max + h_mean
        params = self.pool_proj(h)
        mean, logvar = params.chunk(2, dim=-1)
        return mean, logvar


# ==============================================================================
# Unified Decoder
# ==============================================================================

class UnifiedDecoder(nn.Module):
    """Decode latent vectors to any modality.

    Uses the same architecture for all modalities,
    with modality-specific output heads.
    """

    def __init__(self, latent_dim: int = 512, out_channels: int = 3, channels: list = None):
        super().__init__()
        # 4 upsample layers to match encoder's 4 downsample layers
        channels = channels or [1024, 512, 256, 128, 64]

        # Shared 2D upsample layers (for image/video spatial upsampling)
        layers = [nn.Conv2d(latent_dim, channels[0], 3, padding=1)]
        ch = channels[0]
        for out_ch in channels[1:]:
            layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(),
                ResBlock(out_ch),
            ])
            ch = out_ch

        self.shared_layers = nn.Sequential(*layers)

        # Video temporal upsampling (4x temporal: 4 frames -> 16 frames)
        self.video_temporal_upsample = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, latent_dim, (4, 1, 1), stride=(4, 1, 1), padding=(0, 0, 0)),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
        )

        # Audio 1D decoder
        self.audio_upsample = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose1d(latent_dim, 512, 4, stride=2, padding=1), nn.GroupNorm(8, 512), nn.SiLU()),
            nn.Sequential(nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1), nn.GroupNorm(8, 256), nn.SiLU()),
            nn.Sequential(nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU()),
            nn.Sequential(nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU()),
            nn.Sequential(nn.Conv1d(64, 1, 7, padding=3), nn.Tanh()),
        ])

        # Modality-specific output heads
        self.image_head = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        )

        self.video_head = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        )

        # Point cloud decoder (MLP-based)
        self.pointcloud_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 2048 * 6),  # 2048 points x 6 features (xyz + rgb)
        )

    def forward(self, z: torch.Tensor, modality: str = "image") -> torch.Tensor:
        if modality == "audio":
            z_1d = z.squeeze(2) if z.dim() == 4 else z
            h = z_1d
            for layer in self.audio_upsample:
                h = layer(h)
            return h

        if modality == "video":
            B, C, T, H, W = z.shape
            z_up = self.video_temporal_upsample(z)
            B, C, T_up, H, W = z_up.shape
            z_2d = z_up.permute(0, 2, 1, 3, 4).reshape(B * T_up, C, H, W)
            h = self.shared_layers(z_2d)
            h = self.video_head(h)
            _, Co, Ho, Wo = h.shape
            return h.reshape(B, T_up, Co, Ho, Wo).permute(0, 2, 1, 3, 4)

        if modality == "point_cloud":
            B = z.shape[0]
            z_flat = z.view(B, -1) if z.dim() > 2 else z
            points = self.pointcloud_decoder(z_flat)
            return points.view(B, 2048, 6)

        h = self.shared_layers(z)
        return self.image_head(h)


# ==============================================================================
# Unified VAE
# ==============================================================================

class UnifiedVAE(nn.Module):
    """Unified Variational Autoencoder for all modalities.

    Encodes text, image, video, and audio into a shared latent space.
    Decodes latents back to any modality.

    Usage:
        vae = UnifiedVAE(latent_dim=512)

        # Encode images
        mean, logvar = vae.encode(images, modality="image")
        z = vae.reparameterize(mean, logvar)
        recon = vae.decode(z, modality="image")

        # Encode video
        mean, logvar = vae.encode(video, modality="video")
        z = vae.reparameterize(mean, logvar)
        recon = vae.decode(z, modality="video")

        # Encode audio
        mean, logvar = vae.encode(audio, modality="audio")
        z = vae.reparameterize(mean, logvar)
        recon = vae.decode(z, modality="audio")

        # Cross-modal consistency
        z_image = vae.encode(images, modality="image")
        z_video = vae.encode(video, modality="video")
        # z_image and z_video are in the same latent space
    """

    def __init__(self, latent_dim: int = 512, in_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # Modality-specific encoders
        self.image_encoder = ImageEncoder(in_channels, latent_dim)
        self.video_encoder = VideoEncoder(in_channels, latent_dim)
        self.audio_encoder = AudioEncoder(1, latent_dim)
        self.text_encoder = TextEncoder(latent_dim=latent_dim)
        self.pointcloud_encoder = PointCloudEncoder(6, latent_dim)

        # Shared decoder
        self.decoder = UnifiedDecoder(latent_dim, in_channels)

        # Modality embeddings (for cross-modal consistency)
        self.modality_embed = nn.Embedding(5, latent_dim)  # 0=text, 1=image, 2=video, 3=audio, 4=point_cloud

    def encode(self, x: torch.Tensor, modality: str = "image") -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space.

        Args:
            x: Input tensor (shape depends on modality)
            modality: "text", "image", "video", or "audio"

        Returns:
            (mean, logvar) of the latent distribution
        """
        if modality == "image":
            mean, logvar = self.image_encoder(x)
        elif modality == "video":
            mean, logvar = self.video_encoder(x)
        elif modality == "audio":
            mean, logvar = self.audio_encoder(x)
        elif modality == "text":
            # Text encoding returns sequence of latents
            mean = self.text_encoder(x)
            logvar = torch.zeros_like(mean)  # Text is deterministic
            return mean, logvar
        elif modality == "point_cloud":
            mean, logvar = self.pointcloud_encoder(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Add modality embedding for cross-modal consistency
        mod_id = {"text": 0, "image": 1, "video": 2, "audio": 3, "point_cloud": 4}[modality]
        mod_emb = self.modality_embed(torch.tensor(mod_id, device=x.device))
        # Broadcast mod_emb to match mean's shape
        mod_emb = mod_emb.view(1, -1, *([1] * (mean.dim() - 2)))
        mean = mean + mod_emb

        return mean, logvar

    def decode(self, z: torch.Tensor, modality: str = "image") -> torch.Tensor:
        """Decode latent vector to modality.

        Args:
            z: (B, latent_dim, H, W) latent tensor (or (B, latent_dim) for point_cloud)
            modality: "image", "video", "audio", or "point_cloud"

        Returns:
            Decoded tensor in modality space
        """
        return self.decoder(z, modality)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, x: torch.Tensor, modality: str = "image") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> reparameterize -> decode."""
        mean, logvar = self.encode(x, modality)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z, modality)
        return recon, mean, logvar

    def loss(self, x: torch.Tensor, recon: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE loss = reconstruction + KL divergence."""
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + 0.001 * kl_loss

    @torch.no_grad()
    def sample(self, num_samples: int, modality: str = "image", device: str = "cpu") -> torch.Tensor:
        """Sample from the latent space."""
        if modality == "point_cloud":
            z = torch.randn(num_samples, self.latent_dim, device=device)
            return self.decode(z, modality)
        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        return self.decode(z, modality)

    def get_latent_dim(self) -> int:
        """Get the dimension of the shared latent space."""
        return self.latent_dim
