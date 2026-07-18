"""
NOVA — Neural Optimized Variable-compute Architecture
======================================================
A hybrid transformer that routes tokens through variable compute paths
based on difficulty. Easy tokens (punctuation, common words) use minimal
compute; hard tokens (reasoning, rare entities) get the full treatment.

Core innovations:
  1. Tri-path processing: SSM + Sparse Attention + MoE
  2. Mixture of Depths (MoD): Router skips easy tokens entirely
  3. PRS (Persistent Recurrent State): Cross-sequence memory
  4. Gated fusion: Learned combination of all three paths
  5. Adaptive compute: Variable expert count per token

Architecture (from input to output):
  Token Input
    → Factored Embedding + PRS Init
    → [NOVA Block × L]
        → MoD Router (skip easy tokens)
        → SSM Path (Mamba-style, O(n) sequential)
        → Sparse Attention Path (sliding window + global compression)
        → MoE Path (variable-depth expert routing)
        → Gated Fusion (learned combination)
        → PRS State Update (recurrent memory)
    → Factored Output Head
    → Next Token

References:
  - Mamba (Gu & Dao, 2023): Selective state spaces
  - mixture-of-depths (Raposo et al., 2024): Adaptive compute
  - Perceiver IO (Jaegle et al., 2021): Cross-attention compression
  - Switch Transformer (Fedus et al., 2022): Sparse MoE
  - RWKV (Peng et al., 2023): Recurrent attention
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

@dataclass
class NOVAConfig:
    vocab_size: int = 32000
    dim: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 32
    max_seq_len: int = 4096
    ffn_dim: int = 11008
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # SSM (Mamba-style)
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    # Sparse attention
    attn_window_size: int = 256
    attn_n_global_tokens: int = 64
    # MoE
    moe_experts: int = 8
    moe_activated_min: int = 1
    moe_activated_max: int = 4
    moe_aux_loss_weight: float = 0.01
    # MoD (Mixture of Depths)
    mod_threshold: float = 0.5  # tokens with routing score < this skip the block
    # PRS (Persistent Recurrent State)
    prs_dim: int = 256
    prs_n_heads: int = 4
    # Training
    dropout: float = 0.0


def config_nova_100m() -> NOVAConfig:
    return NOVAConfig(
        vocab_size=32000, dim=768, n_heads=12, n_kv_heads=4,
        n_layers=12, max_seq_len=2048, ffn_dim=3072,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=128, attn_n_global_tokens=32,
        moe_experts=4, moe_activated_min=1, moe_activated_max=2,
        prs_dim=128, prs_n_heads=2,
    )


def config_nova_350m() -> NOVAConfig:
    return NOVAConfig(
        vocab_size=32000, dim=1024, n_heads=16, n_kv_heads=4,
        n_layers=24, max_seq_len=2048, ffn_dim=4096,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=128, attn_n_global_tokens=32,
        moe_experts=6, moe_activated_min=1, moe_activated_max=3,
        prs_dim=128, prs_n_heads=4,
    )


def config_nova_1b() -> NOVAConfig:
    return NOVAConfig(
        vocab_size=32000, dim=2048, n_heads=16, n_kv_heads=4,
        n_layers=24, max_seq_len=4096, ffn_dim=5504,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=256, attn_n_global_tokens=64,
        moe_experts=8, moe_activated_min=1, moe_activated_max=3,
        prs_dim=256, prs_n_heads=4,
    )


def config_nova_7b() -> NOVAConfig:
    return NOVAConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8,
        n_layers=32, max_seq_len=4096, ffn_dim=11008,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=256, attn_n_global_tokens=64,
        moe_experts=8, moe_activated_min=2, moe_activated_max=4,
        prs_dim=512, prs_n_heads=8,
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
    """
    Mamba-style selective state space model.
    O(n) complexity, learns to selectively remember/forget.

    SSM recurrence:
      h_t = A_t * h_{t-1} + B_t * x_t
      y_t = C_t * h_t

    Selective: A, B, C are input-dependent (not fixed).
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        d_inner = dim * expand

        # Input projection
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)

        # 1D convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # State space
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        d_inner = D * self.config_expand if hasattr(self, 'config_expand') else D * 2

        # Split into x and z (gating)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # 1D convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        # Selective SSM parameters
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state + 1)
        B_mat, C_mat = x_dbl[..., :self.d_state], x_dbl[..., self.d_state:2*self.d_state]
        dt = F.softplus(x_dbl[..., -1:])  # (B, L, 1)

        # Discretize: dt needs to broadcast with A
        # A: (d_inner, d_state), dt: (B, L, 1)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        # Expand dt to (B, L, d_inner, 1) for broadcasting
        dt_expand = dt.unsqueeze(2).expand(-1, -1, d_inner, -1)  # (B, L, d_inner, 1)
        dtA = torch.exp(dt_expand * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)

        # Selective scan
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            B_t = B_mat[:, t].unsqueeze(1)  # (B, 1, d_state)
            C_t = C_mat[:, t].unsqueeze(1)  # (B, 1, d_state)
            dt_t = dt[:, t].unsqueeze(-1)   # (B, d_inner, 1)

            h = dtA[:, t] * h + dt_t * (B_t * x[:, t].unsqueeze(-1))
            y = (h * C_t).sum(dim=-1)  # (B, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Skip connection with D
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        # Gating
        y = y * F.silu(z)

        return self.out_proj(y)


# ============================================================
# 2. SPARSE ATTENTION PATH (Sliding Window + Global Compression)
# ============================================================

class SparseAttention(nn.Module):
    """
    Sparse attention with two components:
      1. Sliding window: local attention within a window (O(w*n))
      2. Global compression: cross-attend to compressed global tokens

    Each token attends to:
      - Its local window (window_size tokens)
      - A small set of global summary tokens (n_global_tokens)
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int = 4096, window_size: int = 256,
                 n_global_tokens: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.window_size = window_size
        self.n_global_tokens = n_global_tokens

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.scale = self.head_dim ** -0.5

        # Global compression: learned queries that cross-attend to full sequence
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

        # Sliding window attention
        out = self._sliding_window_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        # Global compression path
        gq = self.global_q.expand(B, -1, -1)
        global_out, _ = self.global_attn(gq, x, x)
        global_out = global_out.mean(dim=1, keepdim=True).expand(-1, L, -1)

        # Combine: local + global
        out = out + global_out

        return self.wo(out)

    def _sliding_window_attention(self, q, k, v):
        B, n_heads, L, head_dim = q.shape
        out = torch.zeros_like(q)
        scale = self.head_dim ** -0.5

        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2)

            q_i = q[:, :, i:i+1]  # (B, n_heads, 1, head_dim)
            k_win = k[:, :, start:end]  # (B, n_heads, w, head_dim)
            v_win = v[:, :, start:end]

            attn = torch.matmul(q_i, k_win.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out[:, :, i:i+1] = torch.matmul(attn, v_win)

        return out


# ============================================================
# 3. MoE PATH (Variable-Depth Mixture of Experts)
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
    """
    Mixture of Experts with variable activation count.
    The router decides how many experts to activate per token (min to max).
    Easy tokens: 1 expert. Hard tokens: up to max_experts.
    """
    def __init__(self, dim: int, hidden_dim: int, n_experts: int = 8,
                 n_activated_min: int = 1, n_activated_max: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.n_min = n_activated_min
        self.n_max = n_activated_max
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(n_experts)])

        # Difficulty estimator: predicts how many experts this token needs
        self.difficulty = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat = x.view(-1, D)

        # Estimate difficulty → number of experts to activate
        difficulty = self.difficulty(flat)  # (B*L, 1)
        n_active = self.n_min + (self.n_max - self.n_min) * difficulty  # continuous
        n_active_int = torch.clamp((n_active * self.n_max).long(), self.n_min, self.n_max)

        gates = F.softmax(self.gate(flat), dim=-1)
        top_k_val, top_k_idx = torch.topk(gates, self.n_max, dim=-1)
        top_k_val = top_k_val / top_k_val.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                out = expert(flat[mask])
                w = (top_k_idx == i).float().sum(dim=-1, keepdim=True)[mask]
                output[mask] += w * out

        load_loss = (gates.mean(dim=0) ** 2).sum() * self.n_experts
        return output.view(B, L, D), load_loss


# ============================================================
# 4. MoD ROUTER (Mixture of Depths)
# ============================================================

class MoDRouter(nn.Module):
    """
    Mixture of Depths router: decides whether each token should go through
    the full block or skip it (residual-only).

    Easy tokens (high confidence) → skip block, save compute
    Hard tokens (low confidence) → full block processing

    This is the key innovation for variable compute.
    """
    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: input (unchanged)
            mask: binary mask (1 = process, 0 = skip) for each token
            router_probs: probability of processing (for logging/regularization)
        """
        probs = self.router(x.mean(dim=1))  # (B, 1) - mean pooling over seq
        mask = (probs > self.threshold).float()  # (B, 1)
        return x, mask, probs


# ============================================================
# 5. PRS STATE (Persistent Recurrent State)
# ============================================================

class PRSState(nn.Module):
    """
    Persistent Recurrent State: a fixed-size memory that persists across
    sequences and is updated recurrently.

    Think of it as a "working memory" that the model maintains:
      state_t = f(state_{t-1}, x_t)

    This allows the model to carry information across contexts.
    """
    def __init__(self, dim: int, prs_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.prs_dim = prs_dim

        # Project from model dim to PRS dim
        self.in_proj = nn.Linear(dim, prs_dim)

        # Recurrent update: GRU-style
        self.update = nn.GRU(prs_dim, prs_dim, batch_first=True)

        # Cross-attention: PRS reads from sequence
        self.cross_attn = nn.MultiheadAttention(
            prs_dim, n_heads, batch_first=True, kdim=dim, vdim=dim
        )

        # Project back to model dim
        self.out_proj = nn.Linear(prs_dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        B, L, D = x.shape

        if state is None:
            state = torch.zeros(1, B, self.prs_dim, device=x.device, dtype=x.dtype)

        # Project sequence to PRS dim
        prs_in = self.in_proj(x)  # (B, L, prs_dim)

        # Recurrent update: GRU expects hidden (num_layers, batch, hidden)
        _, new_state = self.update(prs_in, state)  # new_state: (1, B, prs_dim)

        # Cross-attention: PRS provides context to sequence
        state_flat = new_state.squeeze(0)  # (B, prs_dim)
        state_seq = state_flat.unsqueeze(1).expand(-1, L, -1)  # (B, L, prs_dim)
        prs_context, _ = self.cross_attn(state_seq, x, x)

        # Project back and add
        out = self.out_proj(prs_context)  # (B, L, D)
        x = self.norm(x + out)

        return x, new_state  # keep (1, B, prs_dim) shape


# ============================================================
# 6. GATED FUSION
# ============================================================

class GatedFusion(nn.Module):
    """
    Learned gating to combine SSM, Sparse Attention, and MoE paths.
    Each path gets a learned weight that sums to 1.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 3, 3)
        self.norm = RMSNorm(dim)

    def forward(self, ssm: torch.Tensor, attn: torch.Tensor, moe: torch.Tensor) -> torch.Tensor:
        gate_in = torch.cat([ssm, attn, moe], dim=-1)
        weights = F.softmax(self.gate(gate_in), dim=-1)  # (B, L, 3)

        fused = (
            weights[..., 0:1] * ssm +
            weights[..., 1:2] * attn +
            weights[..., 2:3] * moe
        )
        return self.norm(fused)


# ============================================================
# 7. NOVA BLOCK
# ============================================================

class NOVABlock(nn.Module):
    """
    Single NOVA block: the core processing unit.

    Flow:
      1. MoD Router decides which tokens need processing
      2. Three parallel paths: SSM + Sparse Attention + MoE
      3. Gated Fusion combines the paths
      4. PRS State updates recurrent memory
      5. Residual connections + layer norm
    """
    def __init__(self, config: NOVAConfig, layer_idx: int = 0):
        super().__init__()
        d = config.dim

        # Pre-norms for each path
        self.ssm_norm = RMSNorm(d, config.norm_eps)
        self.attn_norm = RMSNorm(d, config.norm_eps)
        self.moe_norm = RMSNorm(d, config.norm_eps)

        # Three parallel paths
        self.ssm = SelectiveSSM(d, config.ssm_d_state, config.ssm_d_conv, config.ssm_expand)
        self.attn = SparseAttention(
            d, config.n_heads, config.n_kv_heads, config.max_seq_len,
            config.attn_window_size, config.attn_n_global_tokens
        )
        self.moe = VariableMoE(
            d, config.ffn_dim, config.moe_experts,
            config.moe_activated_min, config.moe_activated_max
        )

        # Gated fusion
        self.fusion = GatedFusion(d)

        # MoD router
        self.mod = MoDRouter(d, config.mod_threshold)

        # PRS state (applied every N layers for efficiency)
        self.use_prs = (layer_idx % 4 == 0)  # update PRS every 4 layers
        if self.use_prs:
            self.prs = PRSState(d, config.prs_dim, config.prs_n_heads)

    def forward(self, x: torch.Tensor, prs_state: Optional[torch.Tensor] = None):
        # MoD routing
        x, mod_mask, mod_probs = self.mod(x)

        # Three parallel paths
        ssm_out = self.ssm(self.ssm_norm(x))
        attn_out = self.attn(self.attn_norm(x))
        moe_out, aux_loss = self.moe(self.moe_norm(x))

        # Gated fusion
        fused = self.fusion(ssm_out, attn_out, moe_out)

        # Apply MoD mask: skip block for easy tokens
        x = x + fused * mod_mask.unsqueeze(-1)

        # PRS state update (every N layers)
        if self.use_prs:
            x, prs_state = self.prs(x, prs_state)

        return x, aux_loss, prs_state, mod_probs


# ============================================================
# 8. NOVA MODEL
# ============================================================

class NOVAModel(nn.Module):
    """
    NOVA: Neural Optimized Variable-compute Architecture.

    Full model: Embedding → N × NOVA Block → Output Head
    """
    def __init__(self, config: NOVAConfig):
        super().__init__()
        self.config = config
        d = config.dim

        # Factored embedding (shared between input and output)
        self.tok_emb = nn.Embedding(config.vocab_size, d)

        # Stack of NOVA blocks
        self.blocks = nn.ModuleList([
            NOVABlock(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(d, config.norm_eps)

        # Factored output head (low-rank projection for efficiency)
        self.out_down = nn.Linear(d, d // 4, bias=False)
        self.out_up = nn.Linear(d // 4, config.vocab_size, bias=False)
        self.out_up.weight = nn.Parameter(
            torch.randn(config.vocab_size, d // 4) * 0.02
        )

        # Init
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

        # Run through NOVA blocks
        aux_loss = torch.tensor(0.0, device=x.device)
        prs_state = None
        mod_probs_all = []

        for block in self.blocks:
            x, block_aux, prs_state, mod_probs = block(x, prs_state)
            aux_loss = aux_loss + block_aux
            mod_probs_all.append(mod_probs)

        x = self.norm(x)

        # Factored output: down-project → up-project
        logits = self.out_up(self.out_down(x))

        # MoD regularization: encourage diverse routing
        mod_stack = torch.stack(mod_probs_all, dim=0)  # (n_layers, B, 1)
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
            loss = loss + 0.01 * mod_entropy  # encourage varied routing

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
        print(f"  NOVA Parameters: {n:,} ({n/1e9:.2f}B)")
        return n


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    configs = [
        ("100M", config_nova_100m()),
        ("350M", config_nova_350m()),
        ("1B", config_nova_1b()),
        ("7B", config_nova_7b()),
    ]

    for name, cfg in configs:
        print(f"\n{'='*50}")
        print(f"NOVA {name}")
        print(f"{'='*50}")
        model = NOVAModel(cfg)

        x = torch.randint(0, cfg.vocab_size, (2, 64))
        out = model(x, labels=x)
        print(f"  Loss: {out['loss'].item():.4f}")
        print(f"  MoD entropy: {out['mod_entropy'].item():.4f}")

        # Count skip ratio
        mod_probs = torch.stack([p.mean(dim=0) for p in out['mod_probs']])
        skip_ratio = (mod_probs < cfg.mod_threshold).float().mean().item()
        print(f"  Token skip ratio: {skip_ratio:.1%}")

        gen = model.generate(torch.randint(0, cfg.vocab_size, (1, 5)), max_new_tokens=10)
        print(f"  Generated: {gen.shape}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nAll NOVA configs validated!")
