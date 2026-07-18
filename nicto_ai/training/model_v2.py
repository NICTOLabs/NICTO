"""
NICTO v2 — Production Decoder-Only Transformer
================================================
Architecture based on LLaMA 2 / GPT-NeoX with NICTO-specific extensions.

Core components:
  - RMSNorm (pre-norm)
  - RoPE (Rotary Position Embeddings)
  - Grouped Query Attention (GQA)
  - SwiGLU Feed-Forward Network
  - Optional Mixture of Experts (MoE) FFN
  - Optional Looped Reasoning (Ouro-style)

Parameter counts for reference configs:
  100M: dim=768,  n_heads=12, n_kv_heads=4, n_layers=12, ffn=3072
  350M: dim=1024, n_heads=16, n_kv_heads=4, n_layers=24, ffn=4096
  1B:   dim=2048, n_heads=16, n_kv_heads=4, n_layers=24, ffn=5504
  7B:   dim=4096, n_heads=32, n_kv_heads=8, n_layers=32, ffn=11008
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NICTOConfig:
    vocab_size: int = 32000
    dim: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 32
    max_seq_len: int = 4096
    ffn_dim: int = 11008
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # MoE
    use_moe: bool = False
    moe_experts: int = 8
    moe_activated: int = 2
    moe_aux_loss_weight: float = 0.01
    # Looped reasoning (Ouro-style)
    use_looped: bool = False
    looped_max_steps: int = 4


# ============================================================
# PRETRAINED CONFIGS
# ============================================================

def config_100m() -> NICTOConfig:
    """~100M params — for CPU validation and fast iteration."""
    return NICTOConfig(
        vocab_size=32000, dim=768, n_heads=12, n_kv_heads=4,
        n_layers=12, max_seq_len=2048, ffn_dim=3072,
    )


def config_350m() -> NICTOConfig:
    """~350M params — good for testing on a single consumer GPU."""
    return NICTOConfig(
        vocab_size=32000, dim=1024, n_heads=16, n_kv_heads=4,
        n_layers=24, max_seq_len=2048, ffn_dim=4096,
    )


def config_1b() -> NICTOConfig:
    """~1B params — minimum for coherent text generation."""
    return NICTOConfig(
        vocab_size=32000, dim=2048, n_heads=16, n_kv_heads=4,
        n_layers=24, max_seq_len=4096, ffn_dim=5504,
    )


def config_7b() -> NICTOConfig:
    """~7B params — competitive with LLaMA-2 7B. Needs 1x A100 80GB."""
    return NICTOConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8,
        n_layers=32, max_seq_len=4096, ffn_dim=11008,
    )


def config_7b_moe() -> NICTOConfig:
    """~7B total, ~12B active — MoE variant for higher capacity."""
    return NICTOConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8,
        n_layers=32, max_seq_len=4096, ffn_dim=11008,
        use_moe=True, moe_experts=8, moe_activated=2,
    )


def config_7b_looped() -> NICTOConfig:
    """~7B params with 4 looped steps — Ouro-style reasoning."""
    return NICTOConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8,
        n_layers=32, max_seq_len=4096, ffn_dim=11008,
        use_looped=True, looped_max_steps=4,
    )


# ============================================================
# BUILDING BLOCKS
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._max_seq_len:
            self._max_seq_len = seq_len
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys."""
    # q, k: (B, n_heads, L, head_dim)
    # cos, sin: (L, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Shares KV heads across groups of query heads for efficiency.
    7B config: 32 query heads, 8 KV heads = 4 queries per KV head.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 4096):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # repetitions for KV

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(x, L)
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV heads to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA-style)."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Mixture of Experts Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: int, n_experts: int = 8, n_activated: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFFN(dim, hidden_dim) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        gates = F.softmax(self.gate(x.view(-1, D)), dim=-1)
        top_k_val, top_k_idx = torch.topk(gates, self.n_activated, dim=-1)
        top_k_val = top_k_val / top_k_val.sum(dim=-1, keepdim=True)

        output = torch.zeros(B * L, D, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                out = expert(x.view(-1, D)[mask])
                w = (top_k_idx == i).float().sum(dim=-1, keepdim=True)[mask]
                output[mask] += w * out

        # Load balancing loss
        load_loss = (gates.mean(dim=0) ** 2).sum() * self.n_experts
        return output.view(B, L, D), load_loss


class LoopedBlock(nn.Module):
    """
    Looped reasoning block (Ouro-style).
    Applies the same transformer block recurrently with an exit gate.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int,
                 max_seq_len: int, max_steps: int = 4):
        super().__init__()
        self.max_steps = max_steps

        # Shared transformer block (applied recurrently)
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)
        self.post_norm1 = RMSNorm(dim)
        self.post_norm2 = RMSNorm(dim)

        # Exit gate
        self.exit_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        exit_logprobs = []

        for _ in range(self.max_steps):
            h = self.norm1(x)
            h = self.attn(h)
            x = x + self.post_norm1(h)

            h = self.norm2(x)
            h = self.ffn(h)
            x = x + self.post_norm2(h)

            exit_prob = self.exit_gate(x.mean(dim=1))  # (B, 1)
            exit_logprobs.append(torch.log(1.0 - exit_prob + 1e-8))

        return x, torch.cat(exit_logprobs, dim=-1)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """Single transformer block: attention + FFN with pre-norm."""
    def __init__(self, config: NICTOConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.dim, config.norm_eps)
        self.attn = GroupedQueryAttention(
            config.dim, config.n_heads, config.n_kv_heads, config.max_seq_len
        )
        self.norm2 = RMSNorm(config.dim, config.norm_eps)

        if config.use_moe:
            self.ffn = MoEFFN(config.dim, config.ffn_dim, config.moe_experts, config.moe_activated)
        else:
            self.ffn = SwiGLUFFN(config.dim, config.ffn_dim)

        self.is_looped = config.use_looped and not getattr(self, "_is_last_reasoning", False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        x = x + self.attn(h)

        h = self.norm2(x)
        if isinstance(self.ffn, MoEFFN):
            ffn_out, aux_loss = self.ffn(h)
            x = x + ffn_out
            return x, aux_loss
        else:
            x = x + self.ffn(h)
            return x, torch.tensor(0.0, device=x.device)


# ============================================================
# FULL MODEL
# ============================================================

class NICTOModel(nn.Module):
    """
    NICTO v2: Decoder-only transformer with LLaMA-style architecture.

    Architecture:
        Token Embedding + RoPE → N × (Attention + FFN) → RMSNorm → LM Head

    Weight tying: LM head shares weights with token embedding.
    """

    def __init__(self, config: NICTOConfig):
        super().__init__()
        self.config = config
        d = config.dim

        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.drop = nn.Dropout(0.0)  # LLaMA uses no dropout in pretraining

        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Looped reasoning (applied to last N layers)
        if config.use_looped:
            self.looped = LoopedBlock(
                d, config.n_heads, config.n_kv_heads, config.ffn_dim,
                config.max_seq_len, config.looped_max_steps,
            )
            self.looped_weight = nn.Parameter(torch.zeros(1))  # learned gate

        # Final norm
        self.norm = RMSNorm(d, config.norm_eps)

        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Init weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections (GPT-2 / Muon trick)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        self.count_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, L = input_ids.shape
        x = self.drop(self.tok_emb(input_ids))

        # Transformer layers
        aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, layer_aux = layer(x)
            aux_loss = aux_loss + layer_aux

        # Optional looped reasoning on final representations
        exit_entropy = torch.tensor(0.0, device=x.device)
        if self.config.use_looped:
            looped_x, exit_logprobs = self.looped(x)
            w = torch.sigmoid(self.looped_weight)
            x = x * (1 - w) + looped_x * w
            exit_entropy = -(exit_logprobs.exp() * exit_logprobs).sum(dim=-1).mean()

        x = self.norm(x)

        # Loss
        loss = None
        if labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            if self.config.use_moe:
                loss = loss + self.config.moe_aux_loss_weight * aux_loss
            if self.config.use_looped:
                loss = loss + 0.01 * exit_entropy
        else:
            logits = self.lm_head(x)

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "exit_entropy": exit_entropy,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits = self(idx_cond)["logits"][:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def count_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameters: {n:,} ({n/1e9:.2f}B)")
        return n


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    configs = [
        ("100M", config_100m()),
        ("350M", config_350m()),
        ("1B", config_1b()),
        ("7B", config_7b()),
        ("7B-MoE", config_7b_moe()),
        ("7B-Looped", config_7b_looped()),
    ]

    for name, cfg in configs:
        print(f"\n{'='*50}")
        print(f"NICTO {name}")
        print(f"{'='*50}")
        model = NICTOModel(cfg)

        # Quick forward pass
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        out = model(x, labels=x)
        print(f"  Loss: {out['loss'].item():.4f}")

        # Quick generation
        gen = model.generate(torch.randint(0, cfg.vocab_size, (1, 5)), max_new_tokens=10)
        print(f"  Generated: {gen.shape}")

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nAll configs validated!")
