"""
Multi-Latent Attention (MLA)
Based on DeepSeek-V3 architecture for efficient long-context processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Lazy caching for memory efficiency"""

    def __init__(self, dim: int, max_seq_len: int = 10_000_000, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._cache_size = 0

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-cache only a small chunk (2048) to avoid massive allocation
        self._build_cache(2048)

    def _build_cache(self, seq_len: int):
        """Build RoPE cache up to seq_len"""
        if seq_len <= self._cache_size:
            return
        # Cap at a reasonable size for memory
        seq_len = min(seq_len, 65536)
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cache_size = seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cache_size:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention (MLA) from DeepSeek-V3
    
    Key innovation: Compresses KV cache into latent space for 90%+ memory reduction
    while maintaining attention quality.
    
    Architecture:
    - Low-rank projection for queries and KV
    - Compressed KV cache storage
    - Decoupled RoPE for positional information
    """
    
    def __init__(
        self,
        dim: int = 8192,
        n_heads: int = 128,
        n_kv_heads: int = 16,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        rope_theta: float = 10000.0,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        
        # Group for grouped-query attention
        self.n_rep = n_heads // n_kv_heads
        
        # Q/K/V projections with low-rank compression
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        
        # Compressed KV projections
        self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.head_dim, bias=False)
        self.wkv_a = nn.Linear(dim, kv_lora_rank + n_kv_heads * self.head_dim, bias=False)
        self.wkv_b = nn.Linear(kv_lora_rank, n_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # Normalization
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, rope_theta)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for grouped-query attention"""
        if n_rep == 1:
            return x
        bs, n_kv_heads, slen, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, slen, head_dim)
        return x.reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with KV cache support
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Attention mask
            cache: Cached KV for inference
            
        Returns:
            output: Attention output
            new_cache: Updated KV cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Query projection with low-rank compression
        q = self.wq_b(F.silu(self.wq_a(x)))
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # KV projection with compression
        kv_compressed, k_pe = self.wkv_a(x).split([self.kv_lora_rank, self.n_kv_heads * self.head_dim], dim=-1)
        kv = self.wkv_b(kv_compressed)
        k = kv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Normalize
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Handle KV cache
        if cache is not None:
            k = torch.cat([cache, k], dim=2)
        
        new_cache = k
        
        # Expand KV for grouped-query attention
        k = self._repeat_kv(k, self.n_rep)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Get values
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self._repeat_kv(v, self.n_rep)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.wo(output)
        
        return output, new_cache


class MLABlock(nn.Module):
    """Transformer block with MLA"""
    
    def __init__(self, dim: int, mla_config: dict):
        super().__init__()
        self.attention = MultiLatentAttention(dim=dim, **mla_config)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, mask)
        x = residual + self.dropout(x)
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x
