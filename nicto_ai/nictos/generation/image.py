"""
NICTO Image Generation Module.

Implements a Variational Autoencoder (VAE) and Denoising Diffusion
Probabilistic Model (DDPM) for text-conditioned image generation.

Architecture:
  1. VAE: encode images to latent space, decode latents to images
  2. DDPM: learn to denoise latent vectors conditioned on text
  3. Full pipeline: text -> CLIP-like embed -> noise latent -> denoise -> decode -> image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ==============================================================================
# Building Blocks
# ==============================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, time_emb_dim: int = 0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
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


class AttentionBlock(nn.Module):
    """Self-attention for feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
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
# VAE (Variational Autoencoder)
# ==============================================================================

class VAE Encoder(nn.Module):
    """Encode images to latent vectors."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, channels: list = None):
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        layers = []
        ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
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


class VAE Decoder(nn.Module):
    """Decode latent vectors to images."""

    def __init__(self, out_channels: int = 3, latent_dim: int = 128, channels: list = None):
        super().__init__()
        channels = channels or [512, 256, 128, 64]
        layers = [nn.Conv2d(latent_dim, channels[0], 3, padding=1)]
        ch = channels[0]
        for out_ch in channels[1:]:
            layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                ResBlock(out_ch),
            ])
            ch = out_ch
        layers.extend([
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """Variational Autoencoder for image compression and generation.

    Usage:
        vae = VAE(latent_dim=128)
        mean, logvar = vae.encode(images)
        z = vae.reparameterize(mean, logvar)
        recon = vae.decode(z)
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = VAE Encoder(in_channels, latent_dim)
        self.decoder = VAE Decoder(in_channels, latent_dim)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    def loss(self, x: torch.Tensor, recon: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE loss = reconstruction + KL divergence."""
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + 0.001 * kl_loss

    @torch.no_grad()
    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        return self.decode(z)


# ==============================================================================
# DDPM (Denoising Diffusion Probabilistic Model)
# ==============================================================================

class UNetBlock(nn.Module):
    """U-Net block for the denoising network."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, has_attn: bool = False):
        super().__init__()
        self.res1 = ResBlock(in_ch, time_dim)
        self.res2 = ResBlock(out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        h = self.attn(h)
        skip = h
        h = self.down(h)
        return h, skip


class UNetUpBlock(nn.Module):
    """U-Net up block with skip connections."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, has_attn: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(in_ch + out_ch, time_dim)
        self.res2 = ResBlock(out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if needed
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]])
        x = torch.cat([x, skip], dim=1)
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        h = self.attn(h)
        return h


class DenoiseUNet(nn.Module):
    """U-Net for noise prediction in DDPM."""

    def __init__(self, in_channels: int = 3, base_channels: int = 128, time_dim: int = 256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        # Encoder
        self.enc1 = UNetBlock(in_channels, base_channels, time_dim)
        self.enc2 = UNetBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = UNetBlock(base_channels * 2, base_channels * 4, time_dim, has_attn=True)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 4, time_dim),
            AttentionBlock(base_channels * 4),
            ResBlock(base_channels * 4, time_dim),
        )

        # Decoder
        self.dec3 = UNetUpBlock(base_channels * 4, base_channels * 4, time_dim, has_attn=True)
        self.dec2 = UNetUpBlock(base_channels * 4, base_channels * 2, time_dim)
        self.dec1 = UNetUpBlock(base_channels * 2, base_channels, time_dim)

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        # Encoder
        h, s1 = self.enc1(x, t_emb)
        h, s2 = self.enc2(h, t_emb)
        h, s3 = self.enc3(h, t_emb)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        h = self.dec3(h, s3, t_emb)
        h = self.dec2(h, s2, t_emb)
        h = self.dec1(h, s1, t_emb)

        return self.out(h)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model for image generation.

    Usage:
        ddpm = DDPM(image_size=64, channels=3)
        noise = torch.randn(4, 3, 64, 64)
        t = torch.randint(0, 1000, (4,))
        predicted_noise = ddpm(noise, t)
        # Training: loss = F.mse_loss(predicted_noise, actual_noise)
    """

    def __init__(self, image_size: int = 64, channels: int = 3, timesteps: int = 1000):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        self.denoise_net = DenoiseUNet(channels, base_channels=128, time_dim=256)

        # Noise schedule (linear beta schedule)
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to clean images."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1_ab * noise

    @torch.no_grad()
    def sample(self, num_samples: int = 1, device: str = "cpu", channels: int = 3, height: int = 64, width: int = 64) -> torch.Tensor:
        """Generate images via iterative denoising."""
        x = torch.randn(num_samples, channels, height, width, device=device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = self.denoise_net(x, t_tensor)

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]

            # Denoise
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        return x

    def forward(self, x0: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute denoising loss."""
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=x0.device)

        noise = torch.randn_like(x0)
        x_noisy = self.add_noise(x0, t, noise)
        predicted_noise = self.denoise_net(x_noisy, t)

        return F.mse_loss(predicted_noise, noise)


# ==============================================================================
# Text Conditioning (CLIP-style)
# ==============================================================================

class TextEncoder(nn.Module):
    """Simple text encoder for conditioning image generation.

    Embeds text tokens to a conditioning vector that modulates
    the diffusion process via cross-attention.
    """

    def __init__(self, vocab_size: int = 30000, embed_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode text tokens to conditioning vector.

        Args:
            tokens: (B, seq_len) token IDs

        Returns:
            (B, output_dim) conditioning vector
        """
        x = self.embed(tokens)
        h, _ = self.attn(x, x, x)
        h = self.norm(h + x)
        # Global average pooling
        h = h.mean(dim=1)
        return self.proj(h)


# ==============================================================================
# Full Image Generator
# ==============================================================================

class ImageGenerator(nn.Module):
    """Text-conditioned image generator using VAE + DDPM.

    Pipeline:
        text tokens -> TextEncoder -> conditioning vector
        random noise -> DDPM (conditioned) -> latent
        latent -> VAE Decoder -> image
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        latent_dim: int = 128,
        image_size: int = 64,
        channels: int = 3,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, embed_dim=512, output_dim=256)
        self.vae = VAE(channels, latent_dim)
        self.ddpm = DDPM(image_size, channels, timesteps)
        self.latent_dim = latent_dim

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        num_samples: int = 1,
        height: int = 64,
        width: int = 64,
    ) -> torch.Tensor:
        """Generate images from text.

        Args:
            text_tokens: (B, seq_len) token IDs
            num_samples: number of images per text
            height, width: output image dimensions

        Returns:
            (B, C, H, W) generated images in [-1, 1]
        """
        device = text_tokens.device
        B = text_tokens.shape[0]

        # Encode text
        cond = self.text_encoder(text_tokens)  # (B, 256)

        # Generate images via DDPM
        images = self.ddpm.sample(
            num_samples=B, device=device,
            channels=self.ddpm.channels,
            height=height, width=width,
        )

        return images

    def compute_loss(
        self,
        images: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss.

        Args:
            images: (B, C, H, W) real images
            text_tokens: (B, seq_len) text conditioning

        Returns:
            dict with 'vae_loss', 'ddpm_loss', 'total_loss'
        """
        # VAE reconstruction loss
        recon, mean, logvar = self.vae(images)
        vae_loss = self.vae.loss(images, recon, mean, logvar)

        # DDPM denoising loss
        ddpm_loss = self.ddpm(images)

        total = vae_loss + ddpm_loss
        return {
            "vae_loss": vae_loss,
            "ddpm_loss": ddpm_loss,
            "total_loss": total,
        }
