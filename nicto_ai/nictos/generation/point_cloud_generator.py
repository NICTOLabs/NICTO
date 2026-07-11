"""
3D Point Cloud Generator.

Generates 3D point clouds from text, image, or random latent codes.
Uses PointNet++ style set abstraction for encoding and feature propagation
for decoding. Flow matching for denoising in latent space.

Architecture:
  - PointNet++ Encoder: Set abstraction with multi-scale grouping
  - Flow Matching DiT: Denoise point cloud latents
  - Feature Propagation Decoder: Upsample and predict points + colors
  - Chamfer Distance: Loss for training
  - Cross-Modal: Text/image conditioned generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ==============================================================================
# PointNet++ Building Blocks
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


class SetAbstraction(nn.Module):
    """PointNet++ set abstraction layer (farthest point sampling + grouping)."""

    def __init__(self, in_channels: int, out_channels: int, num_samples: int, group_size: int = 32):
        super().__init__()
        self.num_samples = num_samples
        self.group_size = group_size
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
        )

    def farthest_point_sample(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Farthest point sampling.

        Args:
            points: (B, N, 3) xyz coordinates
            num_samples: number of points to sample

        Returns:
            (B, num_samples) indices of sampled points
        """
        B, N, _ = points.shape
        device = points.device

        centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distances = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), device=device)

        for i in range(num_samples):
            centroids[:, i] = farthest
            centroid = points[torch.arange(B, device=device), farthest].unsqueeze(1)
            dist = torch.sum((points - centroid) ** 2, dim=-1)
            distances = torch.min(distances, dist)
            farthest = torch.argmax(distances, dim=-1)

        return centroids

    def group_points(self, points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Group points around centroids.

        Args:
            points: (B, N, C) point features
            indices: (B, S) centroid indices

        Returns:
            (B, C, S, K) grouped features
        """
        B, N, C = points.shape
        S = indices.shape[1]
        K = self.group_size

        # Get centroid positions
        centroids = torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, C))  # (B, S, C)

        # Find K nearest neighbors for each centroid
        dists = torch.cdist(centroids[:, :, :3], points[:, :, :3])  # (B, S, N)
        _, knn_idx = dists.topk(K, dim=-1, largest=False)  # (B, S, K)

        # Gather grouped points
        grouped = torch.gather(
            points.unsqueeze(1).expand(-1, S, -1, -1),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # (B, S, K, C)

        # Relative position
        grouped = grouped - centroids.unsqueeze(2)

        return grouped.permute(0, 3, 1, 2)  # (B, C, S, K)

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            points: (B, N, C) input points with features

        Returns:
            new_points: (B, out_channels, S) abstracted features
            fps_indices: (B, S) farthest point indices
        """
        B = points.shape[0]
        xyz = points[:, :, :3]

        # Farthest point sampling
        fps_idx = self.farthest_point_sample(xyz, self.num_samples)

        # Group around sampled points
        grouped = self.group_points(points, fps_idx)  # (B, C, S, K)

        # MLP on grouped features
        new_points = self.mlp(grouped)  # (B, out_channels, S, K)

        # Max pooling over neighbors
        new_points = new_points.max(dim=-1)[0]  # (B, out_channels, S)

        return new_points, fps_idx


class FeaturePropagation(nn.Module):
    """PointNet++ feature propagation (interpolation + MLP)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, target_points: torch.Tensor,
                source_points: torch.Tensor) -> torch.Tensor:
        """Interpolate features from source to target points.

        Args:
            x: (B, C, S) source features
            target_points: (B, T, 3) target xyz
            source_points: (B, S, 3) source xyz (fps indices)

        Returns:
            (B, T, out_channels) interpolated features
        """
        B, C, S = x.shape
        T = target_points.shape[1]

        # Transpose for interpolation
        x_t = x.permute(0, 2, 1)  # (B, S, C)

        # Find 3 nearest neighbors in source for each target
        dists = torch.cdist(target_points, source_points)  # (B, T, S)
        _, idx = dists.topk(3, dim=-1, largest=False)  # (B, T, 3)

        # Inverse distance weighting
        gathered = torch.gather(
            x_t.unsqueeze(1).expand(-1, T, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # (B, T, 3, C)

        dists_gathered = torch.gather(dists, 2, idx)  # (B, T, 3)
        weights = 1.0 / (dists_gathered + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (B, T, 3)

        interpolated = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, C)

        return self.mlp(interpolated)


# ==============================================================================
# Point Cloud Encoder
# ==============================================================================

class PointCloudEncoder(nn.Module):
    """Encode point clouds to latent vectors.

    Uses PointNet++ set abstraction for hierarchical feature extraction.
    Input: (B, N, 6) - xyz (3) + rgb (3)
    Output: (B, latent_dim) global feature vector
    """

    def __init__(self, in_channels: int = 6, latent_dim: int = 512):
        super().__init__()
        self.sa1 = SetAbstraction(in_channels, 64, num_samples=512, group_size=32)
        self.sa2 = SetAbstraction(64, 128, num_samples=128, group_size=32)
        self.sa3 = SetAbstraction(128, 256, num_samples=32, group_size=32)

        self.proj = nn.Sequential(
            nn.Linear(256, latent_dim * 2),  # mean + logvar
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud to latent distribution.

        Args:
            x: (B, N, 6) point cloud (xyz + rgb)

        Returns:
            (mean, logvar) each (B, latent_dim)
        """
        # Set abstraction layers
        features, _ = self.sa1(x)
        features, _ = self.sa2(features.permute(0, 2, 1))
        features, _ = self.sa3(features.permute(0, 2, 1))

        # Global max pooling
        global_feat = features.max(dim=-1)[0]  # (B, 256)

        # Project to latent
        params = self.proj(global_feat)
        mean, logvar = params.chunk(2, dim=-1)

        return mean, logvar


# ==============================================================================
# Point Cloud Decoder
# ==============================================================================

class PointCloudDecoder(nn.Module):
    """Decode latent vectors to point clouds.

    Uses hierarchical MLP to generate point coordinates and colors.
    Output: (B, N, 6) - xyz (3) + rgb (3)
    """

    def __init__(self, latent_dim: int = 512, num_points: int = 2048):
        super().__init__()
        self.num_points = num_points

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, num_points * 6),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to point cloud.

        Args:
            z: (B, latent_dim) latent vector

        Returns:
            (B, N, 6) point cloud (xyz + rgb)
        """
        B = z.shape[0]
        points = self.net(z)
        return points.view(B, self.num_points, 6)


# ==============================================================================
# Point Cloud DiT Block
# ==============================================================================

class PointCloudDiTBlock(nn.Module):
    """DiT block for point cloud denoising."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, context_dim: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True,
                                                 kdim=context_dim, vdim=context_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, C) point features
            t_emb: (B, C) time/condition embedding
            context: (B, S, D) cross-attention context

        Returns:
            (B, N, C) denoised features
        """
        # AdaLN modulation
        scale, shift, gate = self.adaln(t_emb).unsqueeze(1).chunk(3, dim=-1)

        # Self-attention
        h = self.norm1(x) * (1 + scale) + shift
        h, _ = self.self_attn(h, h, h)
        x = x + gate * h

        # Cross-attention
        if context is not None:
            h = self.norm2(x)
            h, _ = self.cross_attn(h, context, context)
            x = x + h

        # FFN
        h = self.norm3(x)
        h = self.ffn(h)
        x = x + h

        return x


# ==============================================================================
# Point Cloud Generator
# ==============================================================================

class PointCloudGenerator(nn.Module):
    """3D Point Cloud Generator with Flow Matching.

    Generates point clouds from text/image conditioning using
    flow matching in latent space.

    Architecture:
      1. Encode conditioning (text/image) to context
      2. Sample noise in latent space
      3. Denoise with DiT conditioned on context
      4. Decode latent to point cloud

    Usage:
        gen = PointCloudGenerator(latent_dim=512)
        points = gen.generate(num_points=2048, batch_size=4)
        # points: (4, 2048, 6) - xyz + rgb
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 6, context_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder/Decoder
        self.encoder = PointCloudEncoder(6, latent_dim)
        self.decoder = PointCloudDecoder(latent_dim)

        # DiT for flow matching
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, hidden_dim),
        )

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            PointCloudDiTBlock(hidden_dim, num_heads, context_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Text conditioning (simple projection)
        self.text_proj = nn.Linear(512, context_dim)

        # Image conditioning (simple projection)
        self.image_proj = nn.Linear(512, context_dim)

    def encode_condition(self, text_emb: Optional[torch.Tensor] = None,
                         image_emb: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Encode text/image conditioning to context."""
        contexts = []
        if text_emb is not None:
            contexts.append(self.text_proj(text_emb))
        if image_emb is not None:
            contexts.append(self.image_proj(image_emb))
        if contexts:
            return torch.cat(contexts, dim=1)
        return None

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict velocity field for flow matching.

        Args:
            z_noisy: (B, latent_dim) noisy latent
            t: (B,) timestep
            context: (B, S, D) conditioning context

        Returns:
            (B, latent_dim) predicted velocity
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Project input
        h = self.input_proj(z_noisy).unsqueeze(1)  # (B, 1, hidden_dim)

        # DiT blocks
        for block in self.blocks:
            h = block(h, t_emb, context)

        # Project back to latent
        velocity = self.output_proj(h.squeeze(1))

        return velocity

    @torch.no_grad()
    def generate(self, num_points: int = 2048, batch_size: int = 1,
                 text_emb: Optional[torch.Tensor] = None,
                 image_emb: Optional[torch.Tensor] = None,
                 num_steps: int = 20, device: str = "cpu") -> torch.Tensor:
        """Generate point clouds via flow matching ODE.

        Args:
            num_points: number of points in output
            batch_size: batch size
            text_emb: (B, 512) text embedding (optional)
            image_emb: (B, 512) image embedding (optional)
            num_steps: ODE steps (more = better quality)
            device: device to generate on

        Returns:
            (B, num_points, 6) generated point clouds (xyz + rgb)
        """
        # Encode conditioning
        context = self.encode_condition(text_emb, image_emb)

        # Start from noise
        z = torch.randn(batch_size, self.latent_dim, device=device)

        # Euler ODE solver
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i / num_steps, device=device)
            velocity = self.forward(z, t, context)
            z = z + velocity * dt

        # Decode to point cloud
        points = self.decoder(z)

        return points

    def loss(self, point_cloud: torch.Tensor,
             text_emb: Optional[torch.Tensor] = None,
             image_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flow matching training loss.

        Args:
            point_cloud: (B, N, 6) ground truth point cloud
            text_emb: (B, 512) text embedding (optional)
            image_emb: (B, 512) image embedding (optional)

        Returns:
            scalar loss
        """
        B = point_cloud.shape[0]

        # Encode ground truth
        mean, logvar = self.encoder(point_cloud)
        z0 = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

        # Sample timestep and noise
        t = torch.rand(B, device=point_cloud.device)
        noise = torch.randn_like(z0)
        z_noisy = z0 * (1 - t.unsqueeze(-1)) + noise * t.unsqueeze(-1)

        # Target velocity
        velocity_target = noise - z0

        # Predicted velocity
        context = self.encode_condition(text_emb, image_emb)
        velocity_pred = self.forward(z_noisy, t, context)

        # MSE loss
        loss = F.mse_loss(velocity_pred, velocity_target)

        return loss

    def chamfer_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Chamfer distance between two point clouds.

        Args:
            pred: (B, N, 6) predicted points
            target: (B, M, 6) ground truth points

        Returns:
            scalar loss
        """
        # Pred to target
        dist_p2t = torch.cdist(pred[:, :, :3], target[:, :, :3])
        min_p2t = dist_p2t.min(dim=-1)[0].mean()

        # Target to pred
        dist_t2p = torch.cdist(target[:, :, :3], pred[:, :, :3])
        min_t2p = dist_t2p.min(dim=-1)[0].mean()

        return min_p2t + min_t2p
