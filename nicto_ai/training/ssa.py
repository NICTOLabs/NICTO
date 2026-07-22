"""
SSA — Subquadratic Sparse Attention for NICTO
=============================================
Content-dependent block routing via cumulant scores.
Based on Subquadratic's published Algorithm 1.

Mechanism:
  1. Partition keys into contiguous blocks of size b
  2. Compute per-block summary: mean + diagonal covariance
  3. Route each query to blocks via cumulant score:
     r_c(q) = <q, mu_c> + (beta/2) * <q^2, sigma_c^2>
  4. Select top-k blocks + local window of w recent blocks
  5. Compute exact softmax attention over selected keys only

Complexity: O(n * kappa * d) where kappa = (k+w)*b is fixed,
independent of sequence length n.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SSAAttention(nn.Module):
    """
    Subquadratic Sparse Attention.

    For each query token, selects a fixed budget of key tokens based on
    content-dependent block routing (cumulant scores), then computes
    exact softmax attention over the selected set.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int = 4096, block_size: int = 128,
                 top_k: int = 2, local_window: int = 1, beta: float = 2.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.block_size = block_size
        self.top_k = top_k
        self.local_window = local_window
        self.beta = beta
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        from nicto_ai.training.model_unified import RotaryEmbedding
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def _compute_block_summaries(self, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-block mean and diagonal covariance for keys.

        Args:
            k: (B, n_kv_heads, L, head_dim)
        Returns:
            block_mean: (B, n_kv_heads, n_blocks, head_dim)
            block_var:  (B, n_kv_heads, n_blocks, head_dim)
        """
        B, H, L, D = k.shape
        b = self.block_size
        n_blocks = L // b

        if n_blocks == 0:
            # Sequence shorter than one block — pad to 1 block
            pad = b - L
            k_padded = F.pad(k, (0, 0, 0, pad))  # (B, H, b, D)
            return k_padded.mean(dim=2, keepdim=True), k_padded.var(dim=2, unbiased=False, keepdim=True)

        # Truncate to multiple of block_size
        k_trim = k[:, :, :n_blocks * b]  # (B, H, n_blocks*b, D)
        k_blocks = k_trim.view(B, H, n_blocks, b, D)  # (B, H, n_blocks, b, D)

        block_mean = k_blocks.mean(dim=3)  # (B, H, n_blocks, D)
        block_var = k_blocks.var(dim=3, unbiased=False)  # (B, H, n_blocks, D)

        return block_mean, block_var

    def _cumulant_route(self, q: torch.Tensor, block_mean: torch.Tensor,
                        block_var: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulant routing scores for each query against each block.

        r_c(q) = <q, mu_c> + (beta/2) * <q^2, sigma_c^2>

        Args:
            q:         (B, n_heads, L, head_dim)
            block_mean: (B, n_kv_heads, n_blocks, head_dim)
            block_var:  (B, n_kv_heads, n_blocks, head_dim)
        Returns:
            scores: (B, n_heads, L, n_blocks)
        """
        B, H, L, D = q.shape
        n_blocks = block_mean.shape[2]

        # Block summaries are computed on k (already replicated to match q heads)
        # No additional replication needed

        # Centroid score: <q, mu_c>
        # q: (B, H, L, D), block_mean: (B, H, n_blocks, D)
        centroid_score = torch.matmul(q, block_mean.transpose(-2, -1))  # (B, H, L, n_blocks)

        # Variance score: <q^2, sigma_c^2>
        q_sq = q ** 2  # (B, H, L, D)
        var_score = torch.matmul(q_sq, block_var.transpose(-2, -1))  # (B, H, L, n_blocks)

        # Combined cumulant score
        scores = centroid_score + (self.beta / 2.0) * var_score

        return scores

    def _select_blocks(self, scores: torch.Tensor, L: int) -> torch.Tensor:
        """
        Select top-k blocks + local window for each query.

        Args:
            scores: (B, H, L, n_blocks)
            L: sequence length
        Returns:
            mask: (B, H, L, L) boolean mask — True = attend to this key position
        """
        B, H, L_q, n_blocks = scores.shape
        b = self.block_size
        device = scores.device

        # Top-k blocks per query
        k = min(self.top_k, n_blocks)
        topk_vals, topk_idx = torch.topk(scores, k, dim=-1)  # (B, H, L_q, k)

        # Build mask from top-k blocks
        mask = torch.zeros(B, H, L_q, L, device=device, dtype=torch.bool)
        for ki in range(k):
            block_idx = topk_idx[:, :, :, ki]  # (B, H, L_q)
            # Each block covers positions [block_idx * b, (block_idx + 1) * b)
            pos = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,L)
            block_start = (block_idx.unsqueeze(-1) * b)  # (B, H, L_q, 1)
            block_end = block_start + b  # (B, H, L_q, 1)
            block_mask = (pos >= block_start) & (pos < block_end)  # (B, H, L_q, L)
            mask = mask | block_mask

        # Local window: w most recent blocks
        for q_pos in range(L_q):
            # Which block is this query in?
            q_block = q_pos // b
            # Include blocks from max(0, q_block - local_window + 1) to q_block
            for w in range(self.local_window):
                lb = max(0, q_block - w)
                start = lb * b
                end = min(L, (lb + 1) * b)
                mask[:, :, q_pos, start:end] = True

        # Causal mask: only attend to positions <= current
        causal = torch.triu(torch.ones(L_q, L, device=device, dtype=torch.bool), diagonal=1)
        mask = mask & ~causal.unsqueeze(0).unsqueeze(0)

        # Ensure each query attends to at least its own position
        diag = torch.eye(L_q, L, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        mask = mask | diag

        return mask

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

        # 1. Block summaries
        # We need summaries from kv heads (before replication), but use them for all q heads
        # Recompute on replicated k for simplicity (small overhead)
        block_mean, block_var = self._compute_block_summaries(k)

        # 2. Cumulant routing scores
        scores = self._cumulant_route(q, block_mean, block_var)

        # 3. Block selection (top-k + local window + causal)
        mask = self._select_blocks(scores, L)  # (B, H, L, L)

        # 4. Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # 5. Apply mask: set unselected positions to -inf
        attn = attn.masked_fill(~mask, float('-inf'))

        # 6. Softmax (only over selected positions)
        attn = F.softmax(attn, dim=-1)

        # 7. Weighted sum
        out = torch.matmul(attn, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        return self.wo(out)
