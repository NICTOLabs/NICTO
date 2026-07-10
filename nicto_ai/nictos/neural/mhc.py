"""
Manifold-Constrained Hyper-Connections (mHC) for NICTO.

Based on DeepSeek's mHC paper (arXiv:2512.24880):
"mHC: Manifold-Constrained Hyper-Connections"

Standard residual connections:  x_{l+1} = x_l + F(x_l)
Hyper-Connections:              x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
mHC:                            H_res projected to Birkhoff polytope (doubly stochastic)

Key insight: Unconstrained HC matrices cause signal explosion (gain ~3000x in deep nets).
mHC constrains H_res to doubly stochastic matrices via Sinkhorn-Knopp, ensuring:
  1. Spectral norm <= 1 (non-expansive)
  2. Compositional closure (product of DS matrices is DS)
  3. Convex combination interpretation (energy preserved)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def sinkhorn_knopp(
    matrix: torch.Tensor, num_iters: int = 20, eps: float = 1e-6
) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm to project a matrix onto the Birkhoff polytope.

    Iteratively normalizes rows and columns to sum to 1, producing a
    doubly stochastic matrix from any non-negative input.

    Args:
        matrix: (..., n, n) input matrix (will be exponentiated first)
        num_iters: number of normalization iterations (20 in DeepSeek paper)
        eps: small constant for numerical stability

    Returns:
        (..., n, n) doubly stochastic matrix
    """
    # Start from positive matrix via exponentiation
    M = torch.exp(matrix)

    for _ in range(num_iters):
        # Row normalization
        row_sum = M.sum(dim=-1, keepdim=True).clamp(min=eps)
        M = M / row_sum
        # Column normalization
        col_sum = M.sum(dim=-2, keepdim=True).clamp(min=eps)
        M = M / col_sum

    return M


class ManifoldConstrainedHyperConnections(nn.Module):
    """Manifold-Constrained Hyper-Connections layer.

    Expands the residual stream from C to n*C dimensions and uses
    doubly stochastic matrices for stable signal propagation.

    Forward pass:
        x_flat = flatten(x)  # (B, T, n*C)
        x_normed = RMSNorm(x_flat)

        # Dynamic mappings (input-dependent)
        H_pre  = sigmoid(alpha_pre * (x_normed @ phi_pre) + b_pre)    # (1, n)
        H_post = 2 * sigmoid(alpha_post * (x_normed @ phi_post) + b_post)  # (1, n)
        H_res  = Sinkhorn-Knopp(alpha_res * mat(x_normed @ phi_res) + b_res)  # (n, n)

        # Apply mappings
        layer_input = H_pre @ x_stream           # (B, T, C)
        layer_output = F(layer_input)             # (B, T, C)
        x_new = H_res @ x_stream + H_post^T @ layer_output  # (B, T, n*C)
    """

    def __init__(
        self,
        dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        gating_init: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.n = expansion_rate  # n = expansion rate
        self.nC = dim * expansion_rate
        self.sinkhorn_iters = sinkhorn_iters

        # Linear projections from flattened stream to mapping coefficients
        # phi: (nC) -> (n^2 + 2n) for H_res, H_pre, H_post
        total_coeffs = self.n * self.n + 2 * self.n  # n^2 + 2n
        self.phi = nn.Linear(self.nC, total_coeffs, bias=False)

        # Learnable biases
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))

        # Gating factors (initialized small as per paper)
        self.alpha_res = nn.Parameter(torch.tensor(gating_init))
        self.alpha_pre = nn.Parameter(torch.tensor(gating_init))
        self.alpha_post = nn.Parameter(torch.tensor(gating_init))

        # RMSNorm on the flattened stream
        self.norm = nn.RMSNorm(self.nC, elementwise_affine=False)

    def _compute_mappings(
        self, x_stream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute H_pre, H_post, H_res from the input stream.

        Args:
            x_stream: (B, T, n*C) expanded residual stream

        Returns:
            H_pre:  (B, T, 1, n)
            H_post: (B, T, 1, n)
            H_res:  (B, T, n, n) — doubly stochastic
        """
        B, T, _ = x_stream.shape
        n, nC = self.n, self.dim

        # Normalize and project
        x_normed = self.norm(x_stream)  # (B, T, nC)
        coeffs = self.phi(x_normed)  # (B, T, n^2 + 2n)

        # Split into H_res, H_pre, H_post components
        h_res_flat = coeffs[..., : n * n]  # (B, T, n^2)
        h_pre_flat = coeffs[..., n * n : n * n + n]  # (B, T, n)
        h_post_flat = coeffs[..., n * n + n :]  # (B, T, n)

        # Reshape H_res
        h_res_mat = h_res_flat.reshape(B, T, n, n)  # (B, T, n, n)

        # Apply scaling
        h_res_tilde = self.alpha_res * h_res_mat + self.b_res
        h_pre_tilde = self.alpha_pre * h_pre_flat + self.b_pre
        h_post_tilde = self.alpha_post * h_post_flat + self.b_post

        # H_pre: sigmoid (non-negative)
        H_pre = torch.sigmoid(h_pre_tilde)  # (B, T, 1, n)

        # H_post: 2 * sigmoid (non-negative, scaled)
        H_post = 2.0 * torch.sigmoid(h_post_tilde)  # (B, T, 1, n)

        # H_res: Sinkhorn-Knopp -> doubly stochastic
        H_res = sinkhorn_knopp(h_res_tilde, num_iters=self.sinkhorn_iters)  # (B, T, n, n)

        return H_pre, H_post, H_res

    def forward(
        self,
        x_stream: torch.Tensor,
        layer_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_stream: (B, T, n*C) expanded residual stream
            layer_fn: callable (B, T, C) -> (B, T, C) for the inner layer
                      If None, returns x_stream unchanged (identity test)

        Returns:
            (B, T, n*C) updated stream
        """
        B, T, _ = x_stream.shape
        n, C = self.n, self.dim

        # Compute doubly stochastic mappings
        H_pre, H_post, H_res = self._compute_mappings(x_stream)

        # H_pre: aggregate stream -> layer input
        # x_stream: (B, T, n*C) -> (B, T, n, C)
        x_reshaped = x_stream.reshape(B, T, n, C)
        # H_pre: (B, T, 1, n) -> squeeze to (B, T, n)
        H_pre_squeezed = H_pre.squeeze(2)  # (B, T, n)
        layer_input = torch.einsum("btn,btnc->btc", H_pre_squeezed, x_reshaped)  # (B, T, C)

        # Apply layer function
        if layer_fn is not None:
            layer_output = layer_fn(layer_input)  # (B, T, C)
        else:
            layer_output = layer_input

        # H_res: mix streams
        # x_reshaped: (B, T, n, C), H_res: (B, T, n, n)
        # H_res @ x_reshaped along the n dimension
        res_mixed = torch.einsum("btnm,btmc->btnc", H_res, x_reshaped)  # (B, T, n, C)

        # H_post^T @ layer_output
        # H_post: (B, T, 1, n), layer_output: (B, T, C)
        # We need to broadcast: layer_output contributes to all n streams
        # H_post_squeezed: (B, T, n) -> (B, T, n, 1) to multiply with (B, T, n, C)
        H_post_squeezed = H_post.squeeze(2)  # (B, T, n)
        post_term = layer_output.unsqueeze(2) * H_post_squeezed.unsqueeze(-1)  # (B, T, n, C)

        # Residual merge
        x_new = res_mixed + post_term  # (B, T, n, C)

        return x_new.reshape(B, T, n * C)


class MHCBlock(nn.Module):
    """Transformer block with mHC replacing standard residual connections.

    Uses doubly stochastic residual mixing for stable deep training.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.n = expansion_rate

        # mHC for attention
        self.mhc_attn = ManifoldConstrainedHyperConnections(
            dim, expansion_rate, sinkhorn_iters
        )
        # mHC for FFN
        self.mhc_ffn = ManifoldConstrainedHyperConnections(
            dim, expansion_rate, sinkhorn_iters
        )

        # Attention (operates on C-dim layer input)
        self.attn_norm = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # FFN (operates on C-dim layer input)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard multi-head self-attention on C-dim input."""
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Standard SwiGLU-style FFN on C-dim input."""
        return self.ffn_down(torch.nn.functional.silu(self.ffn_up(x)))

    def forward(self, x_stream: torch.Tensor) -> torch.Tensor:
        """Forward pass through mHC block.

        Args:
            x_stream: (B, T, n*C) expanded residual stream

        Returns:
            (B, T, n*C) updated stream
        """
        # Attention with mHC
        x_stream = self.mhc_attn(x_stream, layer_fn=self._attention)

        # FFN with mHC
        x_stream = self.mhc_ffn(x_stream, layer_fn=self._ffn)

        return x_stream


class MHCModel(nn.Module):
    """Full model with Manifold-Constrained Hyper-Connections.

    Replaces standard residual connections with doubly stochastic
    mixing for stable training at depth.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.n = expansion_rate

        # Token embedding -> expanded stream
        self.embed = nn.Embedding(vocab_size, dim)
        self.stream_proj = nn.Linear(dim, dim * expansion_rate)

        # Transformer blocks with mHC
        self.layers = nn.ModuleList([
            MHCBlock(dim, num_heads, ffn_dim, expansion_rate, sinkhorn_iters)
            for _ in range(num_layers)
        ])

        # Final projection
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (B, T) token IDs

        Returns:
            (B, T, vocab_size) logits
        """
        B, T = token_ids.shape
        n, C = self.n, self.dim

        # Embed and expand to stream
        x = self.embed(token_ids)  # (B, T, C)
        x_stream = self.stream_proj(x)  # (B, T, n*C)

        # Process through mHC layers
        for layer in self.layers:
            x_stream = layer(x_stream)

        # Extract final representation (average over streams)
        x_final = x_stream.reshape(B, T, n, C).mean(dim=2)  # (B, T, C)
        x_final = self.final_norm(x_final)

        return self.head(x_final)
