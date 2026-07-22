"""
ResonanceAttention — Invented for NICTO
========================================
Tokens that resonate with the dominant local context pattern get boosted
attention; tokens that clash get suppressed.

Core idea:
  Standard attention asks "which tokens are relevant to me?"
  ResonanceAttention asks "which tokens share my rhythm?"

  Tokens that align with the same dominant pattern in their local window
  amplify each other, creating emergent functional clusters (syntax groups,
  semantic families, coreference chains) without explicit supervision.

Mechanism:
  1. Compute standard Q/K attention scores
  2. In each local window, compute the dominant pattern (mean of key vectors)
  3. Compute resonance score: cosine similarity of each token's key
     to the dominant pattern
  4. Resonance gate (learned sigmoid): g = sigma(W_r * [resonance, entropy])
  5. Modulated attention: A' = A * (1 + g * resonance)

Why it's novel:
  - Existing attention: weights from Q*K only
  - Gated attention: gates from Q/K, but applied uniformly
  - ResonanceAttention: gates from relationship between token and
    local context structure — a fundamentally different signal
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResonanceAttention(nn.Module):
    """
    ResonanceAttention: attention modulated by local context resonance.

    Each token's attention weights are boosted or suppressed based on how
    well its key aligns with the dominant pattern in its local context window.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int = 4096, resonance_window: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.resonance_window = resonance_window
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        from nicto_ai.training.model_unified import RotaryEmbedding
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        # Resonance gate: learns when to boost/suppress based on
        # [resonance_score, attention_entropy]
        self.gate_proj = nn.Sequential(
            nn.Linear(2, dim // 8),
            nn.SiLU(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid(),
        )

    def _compute_dominant_patterns(self, k: torch.Tensor) -> torch.Tensor:
        """
        Compute dominant pattern (local mean of keys) for each position.

        For position i, the dominant pattern is the mean of keys in
        [max(0, i - window//2), min(L, i + window//2)).

        Args:
            k: (B, n_kv_heads, L, head_dim)
        Returns:
            dominant: (B, n_kv_heads, L, head_dim) — local mean of keys
        """
        B, H, L, D = k.shape
        w = self.resonance_window

        # Use cumulative sum for efficient local mean computation
        # Pad sequence
        k_pad = F.pad(k, (0, 0, w // 2, w // 2))  # (B, H, L + w, D)

        # Cumulative sum
        cumsum = k_pad.cumsum(dim=2)  # (B, H, L + w, D)

        # Local mean = (cumsum[i+w] - cumsum[i]) / w
        dominant = (cumsum[:, :, w:] - cumsum[:, :, :-w]) / w  # (B, H, L, D)

        return dominant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L, x.device)
        from nicto_ai.training.model_unified import apply_rope
        q, k = apply_rope(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 1. Standard attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # 2. Dominant patterns in local windows
        dominant = self._compute_dominant_patterns(k)  # (B, H, L, D)

        # 3. Resonance scores: cosine similarity of each key to its
        #    local dominant pattern
        # k: (B, H, L, D), dominant: (B, H, L, D)
        k_norm = F.normalize(k, dim=-1)
        d_norm = F.normalize(dominant, dim=-1)
        resonance = (k_norm * d_norm).sum(dim=-1)  # (B, H, L)
        resonance = resonance.clamp(-1, 1)

        # 4. Attention entropy (measure of how uniform attention is)
        attn_probs = F.softmax(attn, dim=-1)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1)  # (B, H, L)
        # Normalize entropy to [0, 1] range (max entropy = log(L))
        max_entropy = math.log(L) if L > 1 else 1.0
        entropy_norm = entropy / max_entropy  # (B, H, L)

        # 5. Resonance gate
        gate_input = torch.stack([resonance, entropy_norm], dim=-1)  # (B, H, L, 2)
        gate = self.gate_proj(gate_input)  # (B, H, L, 1)

        # 6. Modulate attention: boost tokens that resonate, suppress clashing
        # Resonance per (query, key) pair: use key's resonance score
        # resonance: (B, H, L) — per key position
        resonance_2d = resonance.unsqueeze(2)  # (B, H, 1, L) — broadcast over queries

        modulation = 1.0 + gate * resonance_2d  # (B, H, L, L)
        attn = attn * modulation

        # 7. Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # 8. Softmax + weighted sum
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        return self.wo(out)
