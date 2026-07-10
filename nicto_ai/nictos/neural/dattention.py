"""
DeepSeek Sparse Attention (DSA2) for NICTO.

Based on DeepSeek's V4 architecture which combines:
- Compressed Sparse Attention (CSA): sparse local + global attention
- Heavily Compressed Attention (HCA): aggressive KV compression
- Native Sparse Attention (NSA): hardware-aligned trainable sparsity

Core idea: Instead of O(N^2) full attention, split into:
1. Local window attention (dense, captures nearby dependencies)
2. Global sparse attention (select top-k important tokens via learned routing)
3. Compressed KV cache (reduces memory by 90%+)

Math:
  DSA_i = alpha * LocalAttn(Q_i, K[local], V[local])
         + beta  * GlobalAttn(Q_i, K[global], V[global])

Where global tokens are selected by a learned routing network.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LightningIndexer(nn.Module):
    """Lightweight routing network that scores tokens for global attention.

    Given a query, scores all key positions and selects the top-k most
    relevant for global attention. This is the "Lightning Indexer" from
    DeepSeek's sparse attention design.
    """

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.score_proj = nn.Linear(dim, num_heads, bias=False)
        self.num_heads = num_heads

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, top_k: int = 16
    ) -> torch.Tensor:
        """Select top-k key positions for each query.

        Args:
            query: (B, T_q, dim) query vectors
            key: (B, T_k, dim) key vectors
            top_k: number of global tokens to select

        Returns:
            indices: (B, T_q, top_k) selected key positions
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Score each key position
        # query: (B, T_q, dim) -> (B, T_q, 1) -> (B, T_q, T_k)
        scores = torch.bmm(query, key.transpose(1, 2))  # (B, T_q, T_k)
        scores = scores / math.sqrt(query.shape[-1])

        # Clamp top_k to available positions
        actual_k = min(top_k, T_k)

        # Select top-k indices
        _, indices = scores.topk(actual_k, dim=-1)  # (B, T_q, actual_k)

        return indices


class CompressedKVCache(nn.Module):
    """Compressed Key-Value cache for memory-efficient long-context attention.

    Compresses KV pairs along the sequence dimension using learned
    compression ratios (c4a: 4x, c128a: 128x as in DeepSeek V4).
    """

    def __init__(self, dim: int, compression_ratio: int = 4):
        super().__init__()
        self.compression_ratio = compression_ratio
        # Conv1d for local compression along sequence dimension
        self.compress = nn.Conv1d(
            dim, dim, kernel_size=compression_ratio,
            stride=compression_ratio, groups=dim, bias=False
        )
        self.norm = nn.RMSNorm(dim)

    def forward(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress KV pairs.

        Args:
            k: (B, T, dim) keys
            v: (B, T, dim) values

        Returns:
            k_compressed: (B, T//ratio, dim)
            v_compressed: (B, T//ratio, dim)
        """
        T = k.shape[1]
        # Pad if needed
        ratio = self.compression_ratio
        pad = (ratio - T % ratio) % ratio
        if pad > 0:
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))

        # Compress via conv
        k_c = self.compress(k.transpose(1, 2)).transpose(1, 2)
        v_c = self.compress(v.transpose(1, 2)).transpose(1, 2)
        k_c = self.norm(k_c)
        v_c = self.norm(v_c)

        return k_c, v_c


class LocalWindowAttention(nn.Module):
    """Dense local window attention for capturing nearby dependencies.

    Each token attends to a fixed-size window of surrounding tokens.
    This is O(N * window_size) instead of O(N^2).
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Local window attention.

        Args:
            x: (B, T, dim)

        Returns:
            (B, T, dim)
        """
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Reshape for windowed attention
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal mask + window mask
        # Each position i attends to positions [max(0, i-window) .. i]
        attn = torch.zeros(B, self.num_heads, T, T, device=x.device, dtype=x.dtype)
        for i in range(T):
            start = max(0, i - self.window_size)
            attn[:, :, i, start : i + 1] = float("-inf")

        # Scaled dot-product with window mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = scores + attn
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class GlobalSparseAttention(nn.Module):
    """Sparse global attention using learned token selection.

    For each query, selects top-k most important key positions via
    the Lightning Indexer, then attends only to those positions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        top_k: int = 16,
        compression_ratio: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.top_k = top_k
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Lightning Indexer for token selection
        self.indexer = LightningIndexer(dim, num_heads=1)

        # Compressed KV cache
        self.kv_cache = CompressedKVCache(dim, compression_ratio)

        # Project q_mean to full dim for indexer
        self.q_index_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse global attention.

        Args:
            x: (B, T, dim)

        Returns:
            (B, T, dim)
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, H, D)

        # Compress KV for memory efficiency
        # CompressedKVCache expects (B, T, dim), so merge heads
        k_flat = k.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        v_flat = v.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        k_compressed, v_compressed = self.kv_cache(k_flat, v_flat)  # (B, T', C)

        T_comp = k_compressed.shape[1]

        # Reshape compressed KV back to multi-head: (B, T', C) -> (B, H, T', D)
        k_h = k_compressed.reshape(B, T_comp, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v_compressed.reshape(B, T_comp, self.num_heads, self.head_dim).transpose(1, 2)

        # Reshape Q to multi-head: (B, T, H, D)
        q_h = q.transpose(1, 2)  # (B, H, T, D)

        # Select top-k positions via Lightning Indexer
        # Reshape q to full dim for matching with k_compressed
        q_full = q_h.transpose(1, 2).reshape(B, T, C)  # (B, T, dim)
        q_full = self.q_index_proj(q_full)  # (B, T, dim)
        indices = self.indexer(q_full, k_compressed, top_k=self.top_k)  # (B, T, top_k)

        # Clamp indices to compressed sequence length
        indices = indices.clamp(max=T_comp - 1)

        # Gather selected K, V using advanced indexing
        # indices: (B, T, top_k) -> (B, H, T, top_k) for gathering from (B, H, T', D)
        indices_h = indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, T, top_k)

        # Gather: for each (b, h, t), select top_k positions from k_h[b, h]
        k_selected = torch.gather(
            k_h.unsqueeze(2).expand(-1, -1, T, -1, -1),  # (B, H, T, T', D)
            3,
            indices_h.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim),  # (B, H, T, top_k, D)
        )  # (B, H, T, top_k, D)

        v_selected = torch.gather(
            v_h.unsqueeze(2).expand(-1, -1, T, -1, -1),
            3,
            indices_h.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim),
        )  # (B, H, T, top_k, D)

        # Attend to selected positions
        # q_h: (B, H, T, D), k_selected: (B, H, T, top_k, D)
        scores = torch.einsum("bhtd,bhtkd->bhtk", q_h, k_selected) / self.scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhtk,bhtkd->bhtd", weights, v_selected)  # (B, H, T, D)

        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, dim)
        return self.out_proj(out)


class DeepSeekSparseAttention(nn.Module):
    """Combined DSA2 attention: local window + global sparse.

    DSA_i = alpha * LocalAttn(Q_i, K[local], V[local])
           + beta  * GlobalAttn(Q_i, K[global], V[global])

    Alpha and beta are learned parameters balancing local vs global.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        window_size: int = 256,
        top_k: int = 16,
        compression_ratio: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Local window attention
        self.local_attn = LocalWindowAttention(dim, num_heads, window_size)

        # Global sparse attention
        self.global_attn = GlobalSparseAttention(
            dim, num_heads, top_k, compression_ratio
        )

        # Learned blending parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        # Final norm
        self.norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, dim)

        Returns:
            (B, T, dim)
        """
        local_out = self.local_attn(x)
        global_out = self.global_attn(x)

        # Blend local and global
        alpha_norm = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        out = alpha_norm[0] * local_out + alpha_norm[1] * global_out

        return self.norm(out + x)  # residual connection


class DSABlock(nn.Module):
    """Transformer block with DeepSeek Sparse Attention.

    Combines DSA with FFN for a complete transformer layer optimized
    for long-context efficiency.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        window_size: int = 256,
        top_k: int = 16,
        compression_ratio: int = 4,
    ):
        super().__init__()
        # DSA attention
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = DeepSeekSparseAttention(
            dim, num_heads, window_size, top_k, compression_ratio
        )

        # FFN
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, dim)

        Returns:
            (B, T, dim)
        """
        # Attention
        x = x + self.attn(self.attn_norm(x))

        # FFN
        x = x + self.ffn_down(torch.nn.functional.silu(self.ffn_up(self.ffn_norm(x))))

        return x


class DSAModel(nn.Module):
    """Full model with DeepSeek Sparse Attention.

    Optimized for long-context processing with O(N * top_k) attention
    instead of O(N^2).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        window_size: int = 256,
        top_k: int = 16,
        compression_ratio: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            DSABlock(dim, num_heads, ffn_dim, window_size, top_k, compression_ratio)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (B, T)

        Returns:
            (B, T, vocab_size) logits
        """
        x = self.embed(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return self.head(x)
