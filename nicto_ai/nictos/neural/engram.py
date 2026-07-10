"""
Engram Conditional Memory — O(1) hash-based memory lookup for NICTO.

Inspired by DeepSeek's Engram paper: "Conditional Memory via Scalable Lookup:
A New Axis of Sparsity for Large Language Models" (arXiv:2601.07372).

The Engram module separates static knowledge retrieval from dynamic reasoning.
It uses deterministic O(1) hash-based lookups for static patterns while
reserving compute for genuine reasoning tasks.

Core idea: N-gram token sequences are hashed to embedding table indices,
retrieved in O(1), gated against the hidden state, and fused back.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional


class NgramHasher:
    """Deterministic N-gram hash function using multiplicative hashing + XOR.

    Given input token IDs, produces hash indices for N-grams of size 2..max_n.
    Each hash head uses a different prime modulus for minimal collision.
    """

    def __init__(
        self,
        vocab_size: int,
        max_ngram: int = 3,
        num_heads: int = 8,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.max_ngram = max_ngram
        self.num_heads = num_heads

        rng = np.random.default_rng(seed)
        max_mult = max(1, (2**31 - 1) // (2 * vocab_size))
        self.multipliers = rng.integers(1, max_mult, size=(max_ngram, num_heads))
        self.moduli = self._generate_moduli(vocab_size, max_ngram, num_heads)

    def _generate_moduli(self, vocab_size: int, max_ngram: int, num_heads: int):
        """Generate prime moduli slightly larger than vocab_size for each head."""
        moduli = np.empty((max_ngram, num_heads), dtype=np.int64)
        base = max(vocab_size, 2)
        candidate = base
        seen = set()
        for n in range(max_ngram):
            for h in range(num_heads):
                while True:
                    candidate += 1
                    if self._is_prime(candidate) and candidate not in seen:
                        break
                moduli[n, h] = candidate
                seen.add(candidate)
        return moduli

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def hash(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute N-gram hash indices.

        Args:
            token_ids: (B, T) token IDs

        Returns:
            (B, T, num_heads * (max_ngram - 1)) hash indices
        """
        B, T = token_ids.shape
        results = []

        for n in range(2, self.max_ngram + 1):
            # Build N-gram: tokens[i-n+1..i] mixed via XOR + multiplication
            mix = token_ids.clone()
            for k in range(1, n):
                shifted = torch.zeros_like(token_ids)
                shifted[:, k:] = token_ids[:, :-k]  # shift right by k
                mix = torch.bitwise_xor(mix, shifted * int(self.multipliers[n - 2][0]))

            for h in range(self.num_heads):
                mod = int(self.moduli[n - 2, h])
                mult = int(self.multipliers[n - 2, h])
                h_idx = torch.bitwise_xor(mix, token_ids * mult) % mod
                results.append(h_idx)

        return torch.stack(results, dim=-1)  # (B, T, num_heads * (max_ngram-1))


class EngramEmbedding(nn.Module):
    """Multi-head embedding table with offset-based partitioning for N-gram heads."""

    def __init__(self, vocab_sizes_per_head: list[int], embed_dim: int):
        super().__init__()
        self.num_heads = len(vocab_sizes_per_head)
        self.embed_dim = embed_dim

        offsets = [0]
        for v in vocab_sizes_per_head[:-1]:
            offsets.append(offsets[-1] + v)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total = sum(vocab_sizes_per_head)
        self.embedding = nn.Embedding(total, embed_dim)

    def forward(self, hash_indices: torch.Tensor) -> torch.Tensor:
        """Embed hash indices with per-head offsets.

        Args:
            hash_indices: (B, T, num_heads) hash values

        Returns:
            (B, T, num_heads, embed_dim) embeddings
        """
        B, T, H = hash_indices.shape
        shifted = hash_indices + self.offsets[:H]
        return self.embedding(shifted)


class EngramGate(nn.Module):
    """Query-key gating between Engram embeddings and hidden states."""

    def __init__(self, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, engram_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute sigmoid gate.

        Args:
            hidden_states: (B, T, hidden_dim)
            engram_embeds: (B, T, num_heads, embed_dim)

        Returns:
            (B, T, num_heads, 1) gate values in [0, 1]
        """
        B, T, H, E = engram_embeds.shape
        key = self.key_proj(engram_embeds)  # (B, T, H, hidden_dim)
        query = self.query_proj(hidden_states).unsqueeze(2)  # (B, T, 1, hidden_dim)

        # Dot product gating with signed square root
        logit = (key * query).sum(dim=-1) / self.scale  # (B, T, H)
        gate = logit.abs().clamp_min(1e-6).sqrt() * logit.sign()
        gate = gate.sigmoid().unsqueeze(-1)  # (B, T, H, 1)
        return gate


class EngramModule(nn.Module):
    """Full Engram Conditional Memory module.

    Retrieves static N-gram knowledge via O(1) hash lookups and fuses
    it with dynamic hidden states via learned gating.

    Architecture:
        1. Hash input tokens to N-gram indices (O(1) per lookup)
        2. Embed via multi-head embedding table
        3. Gate against hidden states
        4. Project gated embeddings back to hidden dim
        5. Add short convolution for local context mixing
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        vocab_size: int = 32000,
        max_ngram: int = 3,
        num_heads: int = 8,
        embed_dim_per_head: int = 64,
        conv_kernel: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim_per_head

        # N-gram hasher (deterministic, no learnable params)
        self.hasher = NgramHasher(
            vocab_size=vocab_size, max_ngram=max_ngram, num_heads=num_heads, seed=seed
        )

        # Per-head vocab sizes (prime numbers > vocab_size)
        vocab_sizes = [int(v) for v in self.hasher.moduli.flatten()[:num_heads]]
        self.embedding = EngramEmbedding(vocab_sizes, embed_dim_per_head)

        # Gating
        self.gate = EngramGate(hidden_dim, embed_dim_per_head)

        # Value projection: per-head embeds -> hidden dim
        self.value_proj = nn.Linear(embed_dim_per_head * num_heads, hidden_dim)

        # Short depthwise conv for local context mixing
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, conv_kernel,
            padding=conv_kernel // 2, groups=hidden_dim, bias=False,
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.act = nn.SiLU()

    def forward(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: (B, T, hidden_dim) current layer input
            token_ids: (B, T) input token IDs

        Returns:
            (B, T, hidden_dim) output with engram memory fused in
        """
        B, T, _ = hidden_states.shape

        # 1. Hash tokens to N-gram indices
        hash_indices = self.hasher.hash(token_ids)  # (B, T, num_heads)
        hash_indices = hash_indices[:, :, : self.num_heads]  # truncate to num_heads

        # 2. Embed via multi-head embedding
        engram_embeds = self.embedding(hash_indices)  # (B, T, num_heads, embed_dim)

        # 3. Gate against hidden states
        gate = self.gate(hidden_states, engram_embeds)  # (B, T, num_heads, 1)

        # 4. Apply gate and project
        gated = gate * engram_embeds  # (B, T, num_heads, embed_dim)
        flat = gated.reshape(B, T, -1)  # (B, T, num_heads * embed_dim)
        value = self.value_proj(flat)  # (B, T, hidden_dim)

        # 5. Short conv for local context
        conv_in = value.transpose(1, 2)  # (B, hidden_dim, T)
        conv_out = self.conv(conv_in)[:, :, :T].transpose(1, 2)
        conv_out = self.act(conv_out)

        # 6. Residual fusion
        output = self.norm(value + conv_out + hidden_states)
        return output


class EngramTransformerBlock(nn.Module):
    """Transformer block with Engram memory augmentation.

    Engram is applied before attention, relieving early layers from
    static pattern reconstruction and preserving depth for reasoning.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        vocab_size: int = 32000,
        max_ngram: int = 3,
        engram_heads: int = 8,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Engram memory
        self.engram = EngramModule(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_ngram=max_ngram,
            num_heads=engram_heads,
            embed_dim_per_head=embed_dim,
        )

        # Standard attention (simplified)
        self.attn_norm = nn.RMSNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # FFN
        self.ffn_norm = nn.RMSNorm(hidden_dim)
        self.ffn_up = nn.Linear(hidden_dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.Tensor):
        """Forward pass through Engram-augmented block.

        Args:
            hidden_states: (B, T, hidden_dim)
            token_ids: (B, T)

        Returns:
            (B, T, hidden_dim)
        """
        # Engram memory retrieval
        h = self.engram(hidden_states, token_ids)

        # Self-attention
        B, T, _ = h.shape
        x = self.attn_norm(h)
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, self.hidden_dim)
        h = h + self.out_proj(attn)

        # FFN
        x = self.ffn_norm(h)
        x = self.ffn_down(torch.nn.functional.silu(self.ffn_up(x)))
        h = h + x

        return h
