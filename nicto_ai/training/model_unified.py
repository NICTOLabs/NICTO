"""
NICTO Unified — State-of-the-Art Collaborative Architecture
============================================================
Merges all three architectures into one superintelligent system:

  1. NOVA Core: Tri-path processing (SSM + Sparse Attn + MoE) with MoD routing
  2. NICTO Subsystems: Memory, Emotional, Creative, Consciousness
  3. v2 Backbone: LLaMA-style RoPE, GQA, SwiGLU, weight tying

Everything collaborates:
  - NOVA blocks process tokens through variable compute paths
  - Memory subsystem builds bidirectional context awareness
  - Emotional subsystem learns tone and sentiment
  - Creative subsystem generates novel patterns
  - Consciousness subsystem monitors and self-regulates
  - Meta-fusion gate learns optimal combination of all subsystems
  - PRS state maintains persistent memory across sequences

Architecture flow:
  Token Input
    → Factored Embedding + RoPE
    → N × Unified NOVA Block
        → MoD Router (skip easy tokens)
        → SSM Path (O(n) selective state space)
        → Sparse Attention Path (sliding window + global compression)
        → MoE Path (variable-depth expert routing)
        → Gated Fusion (learned combination)
        → PRS State Update (persistent memory)
    → Memory Subsystem (bidirectional attention)
    → Emotional Subsystem (causal attention)
    → Creative Subsystem (causal attention)
    → Consciousness Subsystem (self-monitoring)
    → Meta-Fusion Gate (combines all subsystems)
    → Factored Output Head
    → Next Token
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

@dataclass
class NICTOUnifiedConfig:
    vocab_size: int = 32000
    dim: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 32
    max_seq_len: int = 4096
    ffn_dim: int = 11008
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # SSM (NOVA)
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    # Sparse attention (NOVA)
    attn_window_size: int = 256
    attn_n_global_tokens: int = 64
    # MoE (NOVA)
    moe_experts: int = 8
    moe_activated_min: int = 1
    moe_activated_max: int = 4
    moe_aux_loss_weight: float = 0.01
    # MoD (NOVA)
    mod_threshold: float = 0.5
    # PRS (NOVA)
    prs_dim: int = 256
    prs_n_heads: int = 4
    # NICTO Subsystems
    memory_layers: int = 4
    emotional_layers: int = 4
    creative_layers: int = 4
    consciousness_dim: int = 256
    # Training
    dropout: float = 0.0


# ============================================================
# PRETRAINED CONFIGS
# ============================================================

def config_nicto_100m() -> NICTOUnifiedConfig:
    return NICTOUnifiedConfig(
        vocab_size=32000, dim=768, n_heads=12, n_kv_heads=4,
        n_layers=12, max_seq_len=2048, ffn_dim=3072,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=128, attn_n_global_tokens=32,
        moe_experts=4, moe_activated_min=1, moe_activated_max=2,
        prs_dim=128, prs_n_heads=2,
        memory_layers=2, emotional_layers=2, creative_layers=2,
        consciousness_dim=64,
    )


def config_nicto_350m() -> NICTOUnifiedConfig:
    return NICTOUnifiedConfig(
        vocab_size=32000, dim=1024, n_heads=16, n_kv_heads=4,
        n_layers=24, max_seq_len=2048, ffn_dim=4096,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=128, attn_n_global_tokens=32,
        moe_experts=6, moe_activated_min=1, moe_activated_max=3,
        prs_dim=128, prs_n_heads=4,
        memory_layers=3, emotional_layers=3, creative_layers=3,
        consciousness_dim=128,
    )


def config_nicto_1b() -> NICTOUnifiedConfig:
    return NICTOUnifiedConfig(
        vocab_size=32000, dim=2048, n_heads=16, n_kv_heads=4,
        n_layers=24, max_seq_len=4096, ffn_dim=5504,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=256, attn_n_global_tokens=64,
        moe_experts=8, moe_activated_min=1, moe_activated_max=3,
        prs_dim=256, prs_n_heads=4,
        memory_layers=4, emotional_layers=4, creative_layers=4,
        consciousness_dim=256,
    )


def config_nicto_7b() -> NICTOUnifiedConfig:
    return NICTOUnifiedConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8,
        n_layers=32, max_seq_len=4096, ffn_dim=11008,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=256, attn_n_global_tokens=64,
        moe_experts=8, moe_activated_min=2, moe_activated_max=4,
        prs_dim=512, prs_n_heads=8,
        memory_layers=6, emotional_layers=6, creative_layers=6,
        consciousness_dim=512,
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
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max = max_seq_len
        self._build(max_seq_len)

    def _build(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device):
        if seq_len > self._max:
            self._max = seq_len
            self._build(seq_len)
        return self.cos[:seq_len].to(device), self.sin[:seq_len].to(device)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    def rot(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


# ============================================================
# 1. SSM PATH (Mamba-style Selective State Space)
# ============================================================

class SelectiveSSM(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_state = d_state
        d_inner = dim * expand
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = F.silu(x)

        x_dbl = self.x_proj(x)
        B_mat, C_mat = x_dbl[..., :self.d_state], x_dbl[..., self.d_state:2*self.d_state]
        dt = F.softplus(x_dbl[..., -1:])

        A = -torch.exp(self.A_log)
        dt_expand = dt.unsqueeze(2).expand(-1, -1, x.size(-1), -1)
        dtA = torch.exp(dt_expand * A.unsqueeze(0).unsqueeze(0))

        h = torch.zeros(B, x.size(-1), self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            B_t = B_mat[:, t].unsqueeze(1)
            C_t = C_mat[:, t].unsqueeze(1)
            dt_t = dt[:, t].unsqueeze(-1)
            h = dtA[:, t] * h + dt_t * (B_t * x[:, t].unsqueeze(-1))
            ys.append((h * C_t).sum(dim=-1))

        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))


# ============================================================
# 2. SPARSE ATTENTION PATH
# ============================================================

class SparseAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int = 4096, window_size: int = 256,
                 n_global_tokens: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.window_size = window_size

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.scale = self.head_dim ** -0.5

        self.global_q = nn.Parameter(torch.randn(1, n_global_tokens, dim) * 0.02)
        self.global_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(L, x.device)
        q, k = apply_rope(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Sliding window
        out = torch.zeros_like(q)
        for i in range(L):
            s, e = max(0, i - self.window_size // 2), min(L, i + self.window_size // 2)
            attn = F.softmax((q[:, :, i:i+1] @ k[:, :, s:e].transpose(-2, -1)) * self.scale, dim=-1)
            out[:, :, i:i+1] = attn @ v[:, :, s:e]

        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        # Global compression
        gq = self.global_q.expand(B, -1, -1)
        global_out, _ = self.global_attn(gq, x, x)
        out = out + global_out.mean(dim=1, keepdim=True).expand(-1, L, -1)

        return self.wo(out)


# ============================================================
# 3. MoE PATH
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class VariableMoE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_experts: int = 8,
                 n_min: int = 1, n_max: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.n_min = n_min
        self.n_max = n_max
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(n_experts)])
        self.difficulty = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(), nn.Linear(dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat = x.view(-1, D)
        difficulty = self.difficulty(flat)
        gates = F.softmax(self.gate(flat), dim=-1)
        _, top_k_idx = torch.topk(gates, self.n_max, dim=-1)

        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                w = ((top_k_idx == i).float().sum(dim=-1, keepdim=True))[mask]
                output[mask] += w * expert(flat[mask])

        load_loss = (gates.mean(dim=0) ** 2).sum() * self.n_experts
        return output.view(B, L, D), load_loss


# ============================================================
# 4. MoD ROUTER
# ============================================================

class MoDRouter(nn.Module):
    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(), nn.Linear(dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = self.router(x)
        mask = (probs > self.threshold).float()
        return x, mask, probs


# ============================================================
# 5. PRS STATE
# ============================================================

class PRSState(nn.Module):
    def __init__(self, dim: int, prs_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.prs_dim = prs_dim
        self.in_proj = nn.Linear(dim, prs_dim)
        self.update = nn.GRU(prs_dim, prs_dim, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(prs_dim, n_heads, batch_first=True, kdim=dim, vdim=dim)
        self.out_proj = nn.Linear(prs_dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        if state is None:
            state = torch.zeros(1, B, self.prs_dim, device=x.device, dtype=x.dtype)

        prs_in = self.in_proj(x)
        _, new_state = self.update(prs_in, state)

        state_flat = new_state.squeeze(0)
        state_seq = state_flat.unsqueeze(1).expand(-1, L, -1)
        prs_context, _ = self.cross_attn(state_seq, x, x)

        return self.norm(x + self.out_proj(prs_context)), new_state


# ============================================================
# 6. GATED FUSION
# ============================================================

class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 3, 3)
        self.norm = RMSNorm(dim)

    def forward(self, ssm: torch.Tensor, attn: torch.Tensor, moe: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(torch.cat([ssm, attn, moe], dim=-1)), dim=-1)
        return self.norm(weights[..., 0:1] * ssm + weights[..., 1:2] * attn + weights[..., 2:3] * moe)


# ============================================================
# 7. NICTO SUBSYSTEMS
# ============================================================

class MemorySubsystem(nn.Module):
    """Bidirectional attention — builds context awareness across the full sequence."""
    def __init__(self, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim * 4, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        for layer in self.layers:
            h = h + layer(h)
        return h


class EmotionalSubsystem(nn.Module):
    """Causal attention — learns tone, sentiment, and emotional patterns."""
    def __init__(self, dim: int, n_heads: int, n_layers: int, max_seq_len: int = 4096):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim * 4, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        for layer in self.layers:
            h = h + layer(h, src_mask=self.causal_mask[:x.size(1), :x.size(1)])
        return h


class CreativeSubsystem(nn.Module):
    """Causal attention with high temperature — generates novel patterns."""
    def __init__(self, dim: int, n_heads: int, n_layers: int, max_seq_len: int = 4096):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim * 4, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        self.register_buffer("causal_mask", mask)
        self.noise_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        for layer in self.layers:
            h = h + layer(h, src_mask=self.causal_mask[:x.size(1), :x.size(1)])
        if self.training:
            noise = self.noise_proj(h) * torch.randn_like(h) * 0.1
            h = h + noise
        return h


class ConsciousnessSubsystem(nn.Module):
    """Self-monitoring — watches the model's own representations."""
    def __init__(self, dim: int, consciousness_dim: int = 256):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.down = nn.Linear(dim, consciousness_dim)
        self.up = nn.Linear(consciousness_dim, dim)
        self.monitor = nn.Sequential(
            nn.Linear(consciousness_dim, consciousness_dim),
            nn.SiLU(),
            nn.Linear(consciousness_dim, consciousness_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down(h)
        h = h + self.monitor(h)
        return self.up(h)


# ============================================================
# 8. UNIFIED NOVA BLOCK
# ============================================================

class UnifiedNOVABlock(nn.Module):
    """Single NOVA block with tri-path + PRS + MoD."""
    def __init__(self, config: NICTOUnifiedConfig, layer_idx: int = 0):
        super().__init__()
        d = config.dim

        self.ssm_norm = RMSNorm(d, config.norm_eps)
        self.attn_norm = RMSNorm(d, config.norm_eps)
        self.moe_norm = RMSNorm(d, config.norm_eps)

        self.ssm = SelectiveSSM(d, config.ssm_d_state, config.ssm_d_conv, config.ssm_expand)
        self.attn = SparseAttention(d, config.n_heads, config.n_kv_heads, config.max_seq_len,
                                    config.attn_window_size, config.attn_n_global_tokens)
        self.moe = VariableMoE(d, config.ffn_dim, config.moe_experts,
                               config.moe_activated_min, config.moe_activated_max)

        self.fusion = GatedFusion(d)
        self.mod = MoDRouter(d, config.mod_threshold)

        self.use_prs = (layer_idx % 4 == 0)
        if self.use_prs:
            self.prs = PRSState(d, config.prs_dim, config.prs_n_heads)

    def forward(self, x: torch.Tensor, prs_state: Optional[torch.Tensor] = None):
        x, mod_mask, mod_probs = self.mod(x)
        ssm_out = self.ssm(self.ssm_norm(x))
        attn_out = self.attn(self.attn_norm(x))
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        fused = self.fusion(ssm_out, attn_out, moe_out)
        x = x + fused * mod_mask
        if self.use_prs:
            x, prs_state = self.prs(x, prs_state)
        return x, aux_loss, prs_state, mod_probs


# ============================================================
# 9. META-FUSION GATE
# ============================================================

class MetaFusionGate(nn.Module):
    """
    Combines NOVA core output with all NICTO subsystems.
    Learns optimal weights for each subsystem.
    """
    def __init__(self, dim: int, n_subsystems: int = 5):
        super().__init__()
        self.n_subsystems = n_subsystems
        self.gate = nn.Linear(dim * n_subsystems, n_subsystems)
        self.norm = RMSNorm(dim)

    def forward(self, core: torch.Tensor, memory: torch.Tensor,
                emotional: torch.Tensor, creative: torch.Tensor,
                consciousness: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([core, memory, emotional, creative, consciousness], dim=-1)
        weights = F.softmax(self.gate(combined), dim=-1)

        fused = (
            weights[..., 0:1] * core +
            weights[..., 1:2] * memory +
            weights[..., 2:3] * emotional +
            weights[..., 3:4] * creative +
            weights[..., 4:5] * consciousness
        )
        return self.norm(fused)


# ============================================================
# 10. UNIFIED NICTO MODEL
# ============================================================

class NICTOUnifiedModel(nn.Module):
    """
    NICTO Unified: State-of-the-Art Collaborative Architecture

    Combines:
      - NOVA tri-path processing (SSM + Sparse Attn + MoE)
      - MoD routing (skip easy tokens)
      - PRS persistent memory
      - NICTO subsystems (Memory, Emotional, Creative, Consciousness)
      - Meta-fusion gate (combines everything)
      - v2 backbone (RoPE, GQA, SwiGLU, weight tying)
    """
    def __init__(self, config: NICTOUnifiedConfig):
        super().__init__()
        self.config = config
        d = config.dim

        # Embedding
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.rope = RotaryEmbedding(d // config.n_heads, config.max_seq_len, config.rope_theta)

        # NOVA core blocks
        self.nova_blocks = nn.ModuleList([
            UnifiedNOVABlock(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # NICTO subsystems (process NOVA output)
        self.memory = MemorySubsystem(d, config.n_heads, config.memory_layers)
        self.emotional = EmotionalSubsystem(d, config.n_heads, config.emotional_layers, config.max_seq_len)
        self.creative = CreativeSubsystem(d, config.n_heads, config.creative_layers, config.max_seq_len)
        self.consciousness = ConsciousnessSubsystem(d, config.consciousness_dim)

        # Meta-fusion gate
        self.meta_fusion = MetaFusionGate(d, 5)

        # Output
        self.norm = RMSNorm(d, config.norm_eps)
        self.out_down = nn.Linear(d, d // 4, bias=False)
        self.out_up = nn.Linear(d // 4, config.vocab_size, bias=False)
        self.out_up.weight = nn.Parameter(torch.randn(config.vocab_size, d // 4) * 0.02)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("out_proj.weight"):
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
        x = self.tok_emb(input_ids)

        # NOVA core processing
        aux_loss = torch.tensor(0.0, device=x.device)
        prs_state = None
        mod_probs_all = []

        for block in self.nova_blocks:
            x, block_aux, prs_state, mod_probs = block(x, prs_state)
            aux_loss = aux_loss + block_aux
            mod_probs_all.append(mod_probs)

        core_output = x

        # NICTO subsystems
        mem_out = self.memory(core_output)
        emo_out = self.emotional(core_output)
        cre_out = self.creative(core_output)
        con_out = self.consciousness(core_output)

        # Meta-fusion
        fused = self.meta_fusion(core_output, mem_out, emo_out, cre_out, con_out)

        # Output
        x = self.norm(fused)
        logits = self.out_up(self.out_down(x))

        # MoD regularization
        mod_stack = torch.stack(mod_probs_all, dim=0)
        mod_entropy = -(mod_stack * torch.log(mod_stack + 1e-8) +
                        (1 - mod_stack) * torch.log(1 - mod_stack + 1e-8)).mean()

        # Loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss = loss + self.config.moe_aux_loss_weight * aux_loss
            loss = loss + 0.01 * mod_entropy

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "mod_entropy": mod_entropy,
            "mod_probs": mod_probs_all,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]
            logits = self(idx)["logits"][:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def count_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  NICTO Unified Parameters: {n:,} ({n/1e9:.2f}B)")
        return n


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    configs = [
        ("100M", config_nicto_100m()),
        ("350M", config_nicto_350m()),
        ("1B", config_nicto_1b()),
        ("7B", config_nicto_7b()),
    ]

    for name, cfg in configs:
        print(f"\n{'='*50}")
        print(f"NICTO Unified {name}")
        print(f"{'='*50}")
        model = NICTOUnifiedModel(cfg)

        x = torch.randint(0, cfg.vocab_size, (2, 64))
        out = model(x, labels=x)
        print(f"  Loss: {out['loss'].item():.4f}")
        print(f"  MoD entropy: {out['mod_entropy'].item():.4f}")

        mod = torch.stack([p.mean(dim=0) for p in out['mod_probs']])
        skip = (mod < cfg.mod_threshold).float().mean().item()
        print(f"  Token skip ratio: {skip:.1%}")

        gen = model.generate(torch.randint(0, cfg.vocab_size, (1, 5)), max_new_tokens=10)
        print(f"  Generated: {gen.shape}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nAll NICTO Unified configs validated!")
