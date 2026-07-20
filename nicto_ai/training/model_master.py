"""
NICTO MASTER — Ultimate Integration of All Components
======================================================
Everything NICTO has ever been, fused into one unified architecture.

Integrated components:
  1. v2 Backbone: RoPE, GQA, SwiGLU, RMSNorm
  2. NOVA Core: SSM + SparseAttn + MoE + MoD + PRS + GatedFusion
  3. Multimodal I/O: VisionEncoder + AudioEncoder + MultimodalProjector
  4. Cognitive Subsystems: Memory, Emotional, Creative, Consciousness
  5. Looped Reasoning: Ouro-style recurrent block with exit gates
  6. NeuralBus: Cross-network attention + priority gating
  7. DeepSearch: Beam-search multi-step reasoning
  8. HierarchicalMemory: Working → Episodic → Semantic → Procedural
  9. Uncertainty Estimation: Self-monitoring + error detection
  10. Meta-Fusion Gate: Learned weights over ALL outputs

Architecture flow:
  Input (text/image/audio)
    → Multimodal Encoders + Projectors
    → Factored Embedding + RoPE
    → [NOVA Core × N layers]
        → MoD Router (skip easy tokens)
        → SSM Path (O(n) state space)
        → Sparse Attention Path (sliding window + global)
        → MoE Path (variable-depth expert routing)
        → Gated Fusion (learned combination)
        → Cross-Attention (text ↔ visual/audio tokens)
        → PRS State Update (persistent memory)
    → Looped Reasoning (recurrent with exit gates)
    → Cognitive Subsystems (parallel):
        → Memory (bidirectional attention)
        → Emotional (causal attention + emotion vector)
        → Creative (causal attention + noise)
        → Consciousness (uncertainty + self-monitor)
        → HierarchicalMemory (working → episodic → semantic)
    → NeuralBus (cross-network priority attention)
    → DeepSearch (beam-search for hard tokens)
    → Meta-Fusion Gate (weights over ALL outputs)
    → Factored Output Head
    → Next Token
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import from existing NICTO components ──
from nicto_ai.training.model_unified import (
    NICTOUnifiedConfig, UnifiedNOVABlock, GatedFusion,
    MemorySubsystem, EmotionalSubsystem, CreativeSubsystem,
    ConsciousnessSubsystem, MetaFusionGate,
    RMSNorm, RotaryEmbedding, apply_rope,
    SelectiveSSM, SparseAttention, VariableMoE, MoDRouter, PRSState,
)

from nicto_ai.training.model_multimodal import (
    VisionEncoder, AudioEncoder, MultimodalProjector, CrossAttention,
)


# ──────────────────────────────────────────────
# MASTER CONFIG
# ──────────────────────────────────────────────

@dataclass
class NICTOMasterConfig:
    # Vocab & dims
    vocab_size: int = 32000
    dim: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 32
    max_seq_len: int = 4096
    ffn_dim: int = 11008
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0

    # NOVA SSM
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2

    # NOVA Sparse Attention
    attn_window_size: int = 256
    attn_n_global_tokens: int = 64

    # NOVA MoE
    moe_experts: int = 8
    moe_activated_min: int = 1
    moe_activated_max: int = 4
    moe_aux_loss_weight: float = 0.01

    # NOVA MoD
    mod_threshold: float = 0.5

    # NOVA PRS
    prs_dim: int = 256
    prs_n_heads: int = 4

    # Cognitive subsystems
    memory_layers: int = 4
    emotional_layers: int = 4
    creative_layers: int = 4
    consciousness_dim: int = 256

    # Looped reasoning
    use_looped: bool = True
    looped_steps: int = 4

    # Multimodal
    image_size: int = 224
    patch_size: int = 16
    vision_dim: int = 768
    vision_layers: int = 6
    vision_heads: int = 8
    audio_mel_bins: int = 80
    audio_max_frames: int = 1024
    audio_dim: int = 256
    audio_layers: int = 4
    cross_attn_every: int = 2

    # NeuralBus
    use_neural_bus: bool = True
    neural_bus_heads: int = 8

    # DeepSearch
    use_deepsearch: bool = True
    deepsearch_depth: int = 3
    deepsearch_beam: int = 4

    # Loss weights
    mod_entropy_weight: float = 0.01
    exit_entropy_weight: float = 0.01

    # Top Model (Cognitive Executive)
    use_top_model: bool = True
    top_model_layers: int = 2

    # Reward System
    use_reward_system: bool = True
    reward_loss_weight: float = 0.05


# ──────────────────────────────────────────────
# PRETRAINED CONFIGS
# ──────────────────────────────────────────────

def config_master_tiny() -> NICTOMasterConfig:
    return NICTOMasterConfig(
        vocab_size=32000, dim=128, n_heads=4, n_kv_heads=2,
        n_layers=2, max_seq_len=2048, ffn_dim=256,
        ssm_d_state=4, ssm_d_conv=2, ssm_expand=2,
        attn_window_size=32, attn_n_global_tokens=4,
        moe_experts=2, moe_activated_min=1, moe_activated_max=2,
        prs_dim=32, prs_n_heads=2,
        memory_layers=1, emotional_layers=1, creative_layers=1,
        consciousness_dim=16, looped_steps=4,
        image_size=32, patch_size=8, vision_dim=32,
        vision_layers=2, vision_heads=2,
        audio_mel_bins=16, audio_max_frames=32, audio_dim=16, audio_layers=1,
        deepsearch_depth=4, deepsearch_beam=4,
        top_model_layers=1,
    )


def config_master_medium() -> NICTOMasterConfig:
    """Medium config — ~30-40M params, trainable on CPU in reasonable time."""
    return NICTOMasterConfig(
        vocab_size=32000, dim=256, n_heads=8, n_kv_heads=4,
        n_layers=6, max_seq_len=2048, ffn_dim=768,
        ssm_d_state=8, ssm_d_conv=3, ssm_expand=2,
        attn_window_size=64, attn_n_global_tokens=16,
        moe_experts=4, moe_activated_min=1, moe_activated_max=2,
        prs_dim=64, prs_n_heads=4,
        memory_layers=2, emotional_layers=2, creative_layers=2,
        consciousness_dim=64, looped_steps=8,
        image_size=64, patch_size=8, vision_dim=64,
        vision_layers=3, vision_heads=4,
        audio_mel_bins=32, audio_max_frames=128, audio_dim=32, audio_layers=2,
        deepsearch_depth=4, deepsearch_beam=4,
        top_model_layers=2,
    )


def config_master_200m() -> NICTOMasterConfig:
    """200M config — largest model that fits in 8GB RAM with AdamW."""
    return NICTOMasterConfig(
        vocab_size=32000, dim=512, n_heads=8, n_kv_heads=4,
        n_layers=12, max_seq_len=2048, ffn_dim=1536,
        ssm_d_state=8, ssm_d_conv=3, ssm_expand=2,
        attn_window_size=64, attn_n_global_tokens=16,
        moe_experts=4, moe_activated_min=1, moe_activated_max=3,
        prs_dim=128, prs_n_heads=4,
        memory_layers=3, emotional_layers=3, creative_layers=3,
        consciousness_dim=128, looped_steps=10,
        cross_attn_every=3,
        deepsearch_depth=4, deepsearch_beam=4,
        top_model_layers=2,
    )


def config_master_100m() -> NICTOMasterConfig:
    return NICTOMasterConfig(
        vocab_size=32000, dim=768, n_heads=12, n_kv_heads=4,
        n_layers=12, max_seq_len=2048, ffn_dim=3072,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=128, attn_n_global_tokens=32,
        moe_experts=4, moe_activated_min=1, moe_activated_max=2,
        prs_dim=128, prs_n_heads=2,
        memory_layers=2, emotional_layers=2, creative_layers=2,
        consciousness_dim=64, looped_steps=8,
        top_model_layers=2,
    )


def config_master_1b() -> NICTOMasterConfig:
    return NICTOMasterConfig(
        vocab_size=32000, dim=2048, n_heads=16, n_kv_heads=4,
        n_layers=24, max_seq_len=4096, ffn_dim=5504,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=256, attn_n_global_tokens=64,
        moe_experts=8, moe_activated_min=1, moe_activated_max=3,
        prs_dim=256, prs_n_heads=4,
        memory_layers=4, emotional_layers=4, creative_layers=4,
        consciousness_dim=256, looped_steps=12,
        top_model_layers=2,
    )


def config_master_7b() -> NICTOMasterConfig:
    return NICTOMasterConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8,
        n_layers=32, max_seq_len=4096, ffn_dim=11008,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=256, attn_n_global_tokens=64,
        moe_experts=8, moe_activated_min=2, moe_activated_max=4,
        prs_dim=512, prs_n_heads=8,
        memory_layers=6, emotional_layers=6, creative_layers=6,
        consciousness_dim=512, looped_steps=16,
        top_model_layers=3,
    )


def config_master_3b() -> NICTOMasterConfig:
    """3B parameter config — target for distillation from medium teacher."""
    return NICTOMasterConfig(
        vocab_size=32000, dim=1536, n_heads=24, n_kv_heads=8,
        n_layers=24, max_seq_len=2048, ffn_dim=4096,
        ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=128, attn_n_global_tokens=32,
        moe_experts=8, moe_activated_min=2, moe_activated_max=4,
        prs_dim=256, prs_n_heads=8,
        memory_layers=4, emotional_layers=4, creative_layers=4,
        consciousness_dim=256, looped_steps=12,
        deepsearch_depth=4, deepsearch_beam=4,
        top_model_layers=2,
    )


def config_master_5t() -> NICTOMasterConfig:
    """5 trillion parameter config — massive MoE with 512 experts."""
    return NICTOMasterConfig(
        vocab_size=32000, dim=8192, n_heads=64, n_kv_heads=8,
        n_layers=96, max_seq_len=16384, ffn_dim=4096,
        ssm_d_state=32, ssm_d_conv=4, ssm_expand=2,
        attn_window_size=512, attn_n_global_tokens=128,
        moe_experts=512, moe_activated_min=4, moe_activated_max=8,
        prs_dim=1024, prs_n_heads=16,
        memory_layers=24, emotional_layers=24, creative_layers=24,
        consciousness_dim=1024, looped_steps=32,
        image_size=224, patch_size=16, vision_dim=1536,
        vision_layers=24, vision_heads=16,
        audio_mel_bins=80, audio_max_frames=2048, audio_dim=1024, audio_layers=8,
        deepsearch_depth=8, deepsearch_beam=16,
        top_model_layers=4,
    )


# ──────────────────────────────────────────────
# 4. LOOPED REASONING (Ouro-style)
# ──────────────────────────────────────────────

class LoopedReasoningBlock(nn.Module):
    """Recurrent reasoning with learned exit gates."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 ffn_dim: int, max_seq_len: int, max_steps: int = 4):
        super().__init__()
        self.max_steps = max_steps
        from nicto_ai.training.model_v2 import GroupedQueryAttention, SwiGLUFFN
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)
        self.post_norm1 = RMSNorm(dim)
        self.post_norm2 = RMSNorm(dim)
        self.exit_gate = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        exit_logprobs = []
        for _ in range(self.max_steps):
            h = self.norm1(x); h = self.attn(h); x = x + self.post_norm1(h)
            h = self.norm2(x); h = self.ffn(h); x = x + self.post_norm2(h)
            exit_prob = self.exit_gate(x.mean(dim=1))
            exit_logprobs.append(torch.log(1.0 - exit_prob + 1e-8))
        return x, torch.cat(exit_logprobs, dim=-1)


# ──────────────────────────────────────────────
# 5. NEURALBUS
# ──────────────────────────────────────────────

class NeuralBus(nn.Module):
    """Cross-network attention + priority gating."""
    def __init__(self, dim: int, n_networks: int = 6, n_heads: int = 8):
        super().__init__()
        self.n_networks = n_networks
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(dim, n_heads, batch_first=True)
            for _ in range(n_networks)
        ])
        self.projections = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_networks)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(dim * n_networks, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        self.priority_gate = nn.Sequential(
            nn.Linear(dim, n_networks), nn.Softmax(dim=-1),
        )

    def forward(self, network_outputs: List[torch.Tensor]) -> torch.Tensor:
        B = network_outputs[0].size(0)
        projected = [p(o.mean(dim=1)) for p, o in zip(self.projections, network_outputs)]
        attended = []
        for i in range(self.n_networks):
            query = network_outputs[i].mean(dim=1, keepdim=True)
            others = torch.stack([
                o.mean(dim=1) for j, o in enumerate(network_outputs) if j != i
            ], dim=1)
            out, _ = self.cross_attn[i](query, others, others)
            attended.append(out.squeeze(1))
        fused = self.fusion(torch.cat(attended, dim=-1))
        gates = self.priority_gate(fused)
        scaled = [projected[i] * gates[:, i:i+1] for i in range(self.n_networks)]
        final = self.fusion(torch.cat(scaled, dim=-1))
        return final


# ──────────────────────────────────────────────
# 6. DEEPSEARCH
# ──────────────────────────────────────────────

class DeepSearchBlock(nn.Module):
    """Beam-search reasoning with multi-step exploration."""
    def __init__(self, dim: int, max_depth: int = 3, beam_width: int = 4):
        super().__init__()
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.think_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.mean(dim=1)
        thoughts = [self.think_proj(x_flat).unsqueeze(1) for _ in range(self.beam_width)]
        for depth in range(self.max_depth):
            candidates = []
            for b in range(self.beam_width):
                h = self.think_proj(thoughts[b])
                score = self.scorer(h)
                candidates.append((h, score))
            sorted_cands = sorted(candidates, key=lambda c: c[1].mean().item(), reverse=True)
            thoughts = [c[0] for c in sorted_cands[:self.beam_width]]
        best = thoughts[0]
        best_seq = best.expand(-1, L, -1)
        return x + best_seq


# ──────────────────────────────────────────────
# 7. HIERARCHICAL MEMORY
# ──────────────────────────────────────────────

class HierarchicalMemoryBlock(nn.Module):
    """Working → Episodic → Semantic memory with consolidation."""
    def __init__(self, dim: int, capacity: int = 1024):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.working_memory = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        self.episodic = nn.Parameter(torch.zeros(1, capacity, dim))
        self.semantic = nn.Parameter(torch.zeros(1, capacity, dim))
        self.write_gate = nn.Linear(dim * 2, dim)
        self.read_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.pointer = nn.Parameter(torch.zeros(1))
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_mean = x.mean(dim=1, keepdim=True)
        working = self.working_memory(x_mean)
        write_idx = int(self.pointer.item()) % self.capacity
        self.pointer.data += 1
        mem = self.episodic.expand(B, -1, -1)
        mem_out, _ = self.read_attn(x, mem, mem)
        return self.norm(x + mem_out)


# ──────────────────────────────────────────────
# 8. UNCERTAINTY ESTIMATOR
# ──────────────────────────────────────────────

class UncertaintyEstimator(nn.Module):
    """Self-monitoring: per-token uncertainty + calibration."""
    def __init__(self, dim: int):
        super().__init__()
        self.per_token_monitor = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid(),
        )
        self.sequence_monitor = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid(),
        )
        self.calibrate = nn.Sequential(
            nn.Linear(1, dim // 8), nn.SiLU(),
            nn.Linear(dim // 8, 1),
        )

    def forward(self, x: torch.Tensor, logits: Optional[torch.Tensor] = None) -> Dict:
        token_unc = self.per_token_monitor(x).squeeze(-1)   # (B, L)
        seq_unc = self.sequence_monitor(x.mean(dim=1))       # (B, 1)
        entropy = None
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            seq_unc = seq_unc + self.calibrate(entropy.detach().unsqueeze(-1))
        combined = (token_unc + seq_unc).clamp(0, 1)
        return {"uncertainty": combined, "entropy": entropy}


# ──────────────────────────────────────────────
# 10. REWARD SYSTEM
# ──────────────────────────────────────────────

class SubsystemRewardHead(nn.Module):
    """Scores a subsystem's proposal on a single reward dimension."""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(),
            nn.Linear(dim // 4, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x.mean(dim=1)).squeeze(-1)  # (B,)


class RewardSystem(nn.Module):
    """
    Multi-dimensional reward scoring for subsystem proposals.

    Each subsystem is scored on 4 dimensions:
      - coherence:   grammatical/semantic correctness
      - relevance:   contextual appropriateness
      - creativity:  novelty and diversity
      - safety:      freedom from harmful content

    The reward system produces:
      - Per-subsystem reward scores (B, n_subsystems)
      - Reward-weighted proposals for the CognitiveExecutive
      - Auxiliary reward loss for training signal
    """
    REWARD_DIMS = ["coherence", "relevance", "creativity", "safety"]

    def __init__(self, dim: int, n_subsystems: int = 5):
        super().__init__()
        self.n_subsystems = n_subsystems

        # Per-subsystem, per-dimension reward heads
        self.reward_heads = nn.ModuleDict({
            name: nn.ModuleList([
                SubsystemRewardHead(dim) for _ in range(n_subsystems)
            ])
            for name in self.REWARD_DIMS
        })

        # Meta-reward combiner: learns to weight reward dimensions
        self.meta_combiner = nn.Sequential(
            nn.Linear(n_subsystems * len(self.REWARD_DIMS), n_subsystems),
            nn.Sigmoid(),
        )

        # Reward-to-weight projection for CognitiveExecutive
        self.reward_to_weight = nn.Sequential(
            nn.Linear(n_subsystems + 1, n_subsystems),
            nn.Softmax(dim=-1),
        )

    def forward(self, subsystem_outputs: List[torch.Tensor],
                uncertainty: torch.Tensor) -> Dict:
        """
        Args:
            subsystem_outputs: list of (B, L, D) from each subsystem
            uncertainty: (B, L) per-token uncertainty
        Returns:
            Dict with:
              - reward_scores: (B, n_subsystems) combined reward per subsystem
              - reward_weights: (B, n_subsystems) attention weights for Top Model
              - per_dim_scores: dict of (B, n_subsystems) per-dimension scores
              - reward_loss: scalar auxiliary loss
        """
        B = subsystem_outputs[0].size(0)

        # Score each subsystem on each dimension
        per_dim_scores = {}
        all_dim_scores = []
        for dim_name in self.REWARD_DIMS:
            scores = []
            for i, head in enumerate(self.reward_heads[dim_name]):
                scores.append(head(subsystem_outputs[i]))
            stacked = torch.stack(scores, dim=1)  # (B, n_subsystems)
            per_dim_scores[dim_name] = stacked
            all_dim_scores.append(stacked)

        # Concatenate all dimension scores
        all_scores = torch.cat(all_dim_scores, dim=-1)  # (B, n_sub * n_dims)

        # Meta-combine to get per-subsystem reward
        reward_scores = self.meta_combiner(all_scores)  # (B, n_subsystems)

        # Compute reward weights using uncertainty
        seq_unc = uncertainty.mean(dim=1)  # (B,)
        reward_input = torch.cat([reward_scores, seq_unc.unsqueeze(-1)], dim=-1)
        reward_weights = self.reward_to_weight(reward_input)  # (B, n_subsystems)

        # Auxiliary loss: encourage diverse but high reward across subsystems
        # Subsystems should not all get the same reward
        reward_variance = reward_scores.var(dim=1).mean()
        reward_mean = reward_scores.mean()
        reward_loss = (1.0 - reward_mean) + 0.1 * (1.0 - reward_variance)

        return {
            "reward_scores": reward_scores,
            "reward_weights": reward_weights,
            "per_dim_scores": per_dim_scores,
            "reward_loss": reward_loss,
        }


# ──────────────────────────────────────────────
# 11. COGNITIVE EXECUTIVE (Top Model)
# ──────────────────────────────────────────────

class CognitiveExecutive(nn.Module):
    """
    The top model — cognitive executive.

    Receives competing token proposals from all subsystems, weighted by
    the RewardSystem, and makes the final decision via cross-attention reasoning.

    Architecture:
      1. Each subsystem has a lightweight proposal head
      2. Proposals are weighted by reward scores
      3. Weighted proposals are concatenated and fused
      4. Uncertainty-aware gating further modulates the signal
      5. Self-attention reasoning layers process the result
      6. Residual connection to meta-fused output
      7. Shared output head produces final logits
    """
    def __init__(self, dim: int, n_heads: int, n_subsystems: int = 5,
                 n_layers: int = 2):
        super().__init__()
        self.n_subsystems = n_subsystems

        # Per-subsystem proposal heads (lightweight linear projections)
        self.proposal_heads = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_subsystems)
        ])

        # Fusion: concatenate weighted proposals → single representation
        self.fusion = nn.Sequential(
            nn.Linear(dim * n_subsystems, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Uncertainty-aware gating
        self.unc_gate = nn.Sequential(
            nn.Linear(dim + 1, dim), nn.Sigmoid(),
        )

        # Reasoning layers (self-attention over fused proposals)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                dim, n_heads, dim * 4,
                activation=F.gelu, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, fused: torch.Tensor,
                subsystem_outputs: List[torch.Tensor],
                uncertainty: torch.Tensor,
                reward_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            fused: (B, L, D) from meta-fusion gate
            subsystem_outputs: list of (B, L, D) from each subsystem
            uncertainty: (B, L) per-token uncertainty
            reward_weights: (B, n_subsystems) from RewardSystem
        Returns:
            (B, L, D) refined representation
        """
        B, L, D = fused.shape

        # Project each subsystem's output through its proposal head
        projected = []
        for i, out in enumerate(subsystem_outputs):
            h = self.proposal_heads[i](out)  # (B, L, D)

            # Apply reward weighting if available
            if reward_weights is not None:
                w = reward_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                h = h * w

            projected.append(h)

        # Concatenate and fuse
        concat = torch.cat(projected, dim=-1)  # (B, L, n_sub * D)
        h = self.fusion(concat)                 # (B, L, D)

        # Uncertainty-aware gating: trust uncertain subsystems less
        unc = uncertainty.unsqueeze(-1)         # (B, L, 1)
        gate = self.unc_gate(torch.cat([h, unc], dim=-1))
        h = h * gate

        # Reasoning layers
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Residual connection to meta-fused output
        h = h + fused

        return h


# ──────────────────────────────────────────────
# 10. MASTER MODEL
# ──────────────────────────────────────────────

class NICTOMasterModel(nn.Module):
    """
    NICTO MASTER — the ultimate NICTO architecture.
    Everything integrated into one unified system.

    Architecture:
      Bottom stack: NOVA → Looped → Subsystems → NeuralBus → DeepSearch → Meta-Fusion
      Top model:    CognitiveExecutive (cross-attention over subsystem proposals)
    """
    def __init__(self, config: NICTOMasterConfig):
        super().__init__()
        self.config = config
        d = config.dim

        # ── Text Embedding ──
        self.tok_emb = nn.Embedding(config.vocab_size + 4, d)
        self.rope = RotaryEmbedding(d // config.n_heads, config.max_seq_len, config.rope_theta)

        # ── Multimodal Encoders ──
        self.vision_encoder = VisionEncoder.__new__(VisionEncoder)
        nn.Module.__init__(self.vision_encoder)
        from nicto_ai.training.model_multimodal import PatchEmbed
        ve = self.vision_encoder
        ve.patch_embed = PatchEmbed(config.image_size, config.patch_size, embed_dim=config.vision_dim)
        num_patches = ve.patch_embed.num_patches
        ve.cls_token = nn.Parameter(torch.randn(1, 1, config.vision_dim) * 0.02)
        ve.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.vision_dim) * 0.02)
        ve.pos_drop = nn.Dropout(p=0.0)
        ve.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(config.vision_dim, config.vision_heads,
                config.vision_dim * 4, activation=F.gelu, batch_first=True, norm_first=True)
            for _ in range(config.vision_layers)
        ])
        ve.norm = nn.LayerNorm(config.vision_dim)
        self.vision_projector = MultimodalProjector(config.vision_dim, d)

        self.audio_encoder = AudioEncoder.__new__(AudioEncoder)
        if hasattr(config, 'audio_mel_bins'):
            nn.Module.__init__(self.audio_encoder)
            self.audio_encoder.mel_bins = config.audio_mel_bins
            self.audio_encoder.conv_stack = nn.Sequential(
                nn.Conv1d(config.audio_mel_bins, config.audio_dim, 3, padding=1), nn.GELU(),
                nn.Conv1d(config.audio_dim, config.audio_dim, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv1d(config.audio_dim, config.audio_dim, 3, stride=2, padding=1), nn.GELU(),
            )
            self.audio_encoder.pos_embed = nn.Parameter(torch.randn(1, config.audio_max_frames // 4, config.audio_dim) * 0.02)
            self.audio_encoder.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(config.audio_dim, 4, config.audio_dim * 4,
                    activation=F.gelu, batch_first=True, norm_first=True)
                for _ in range(config.audio_layers)
            ])
            self.audio_encoder.norm = nn.LayerNorm(config.audio_dim)
        self.audio_projector = MultimodalProjector(config.audio_dim, d) if hasattr(config, 'audio_dim') else None

        # ── NOVA Core Blocks ──
        self.nova_blocks = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        for i in range(config.n_layers):
            block = UnifiedNOVABlock(
                self._make_nova_config(config), layer_idx=i
            )
            self.nova_blocks.append(block)
            ca = (CrossAttention(d, config.n_heads)
                  if (i % config.cross_attn_every == 0) else nn.Identity())
            self.cross_attn_layers.append(ca)

        # ── Looped Reasoning ──
        if config.use_looped:
            self.looped = LoopedReasoningBlock(d, config.n_heads, config.n_kv_heads,
                config.ffn_dim, config.max_seq_len, config.looped_steps)
            self.looped_gate = nn.Parameter(torch.zeros(1))

        # ── Cognitive Subsystems ──
        self.memory = MemorySubsystem(d, config.n_heads, config.memory_layers)
        self.emotional = EmotionalSubsystem(d, config.n_heads, config.emotional_layers, config.max_seq_len)
        self.creative = CreativeSubsystem(d, config.n_heads, config.creative_layers, config.max_seq_len)
        self.consciousness = ConsciousnessSubsystem(d, config.consciousness_dim)
        self.hierarchical_memory = HierarchicalMemoryBlock(d)

        # ── NeuralBus ──
        if config.use_neural_bus:
            self.neural_bus = NeuralBus(d, n_networks=5, n_heads=config.neural_bus_heads)

        # ── DeepSearch ──
        if config.use_deepsearch:
            self.deepsearch = DeepSearchBlock(d, config.deepsearch_depth, config.deepsearch_beam)

        # ── Uncertainty Estimator ──
        self.uncertainty = UncertaintyEstimator(d)

        # ── Meta-Fusion Gate ──
        self.meta_fusion = MetaFusionGate(d, 5)

        # ── Top Model (Cognitive Executive) ──
        self.top_model = None
        self.reward_system = None
        if config.use_top_model:
            self.top_model = CognitiveExecutive(
                d, config.n_heads, n_subsystems=5,
                n_layers=config.top_model_layers,
            )
            if config.use_reward_system:
                self.reward_system = RewardSystem(d, n_subsystems=5)

        # ── Output ──
        self.norm = RMSNorm(d, config.norm_eps)
        self.out_down = nn.Linear(d, d // 4, bias=False)
        self.out_up = nn.Linear(d // 4, config.vocab_size, bias=False)
        self.out_up.weight = nn.Parameter(torch.randn(config.vocab_size, d // 4) * 0.02)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        self.count_parameters()

    def _make_nova_config(self, c):
        ncfg = NICTOUnifiedConfig.__new__(NICTOUnifiedConfig)
        for attr in ['vocab_size', 'dim', 'n_heads', 'n_kv_heads', 'n_layers',
                     'max_seq_len', 'ffn_dim', 'norm_eps', 'rope_theta',
                     'ssm_d_state', 'ssm_d_conv', 'ssm_expand',
                     'attn_window_size', 'attn_n_global_tokens',
                     'moe_experts', 'moe_activated_min', 'moe_activated_max',
                     'moe_aux_loss_weight', 'mod_threshold',
                     'prs_dim', 'prs_n_heads',
                     'memory_layers', 'emotional_layers', 'creative_layers',
                     'consciousness_dim', 'dropout']:
            setattr(ncfg, attr, getattr(c, attr))
        return ncfg

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        device = next(self.parameters()).device
        all_tokens, modality_mask = [], []

        # Text
        if input_ids is not None:
            B = input_ids.size(0)
            text_emb = self.tok_emb(input_ids)
            all_tokens.append(text_emb)
            modality_mask.append(torch.zeros(B, input_ids.size(1), device=device))

        # Images
        if images is not None:
            B = images.size(0)
            vis = self.vision_encoder(images)
            vis = self.vision_projector(vis)
            all_tokens.append(vis)
            modality_mask.append(torch.ones(B, vis.size(1), device=device))

        # Audio
        if audio is not None and self.audio_projector is not None:
            B = audio.size(0)
            aud = self.audio_encoder(audio)
            aud = self.audio_projector(aud)
            all_tokens.append(aud)
            modality_mask.append(torch.full((B, aud.size(1)), 2, device=device))

        if not all_tokens:
            raise ValueError("At least one input required")

        x = torch.cat(all_tokens, dim=1)
        modality_mask = torch.cat(modality_mask, dim=1)

        # ── NOVA Core Blocks ──
        aux_loss = torch.tensor(0.0, device=x.device)
        prs_state = None
        mod_probs_all = []

        for i, (block, ca) in enumerate(zip(self.nova_blocks, self.cross_attn_layers)):
            x, block_aux, prs_state, mod_probs = block(x, prs_state)
            aux_loss = aux_loss + block_aux
            mod_probs_all.append(mod_probs)
            if isinstance(ca, CrossAttention) and images is not None:
                vis_mask = (modality_mask == 1)
                txt_mask = (modality_mask == 0)
                if vis_mask.any() and txt_mask.any():
                    for b in range(x.size(0)):
                        if txt_mask[b].any() and vis_mask[b].any():
                            x[b:b+1, txt_mask[b]] = ca(
                                x[b:b+1, txt_mask[b]], x[b:b+1, vis_mask[b]]
                            )

        core_output = x

        # ── Looped Reasoning ──
        exit_entropy = torch.tensor(0.0, device=x.device)
        if self.config.use_looped:
            looped_out, exit_logprobs = self.looped(core_output)
            w = torch.sigmoid(self.looped_gate)
            core_output = core_output * (1 - w) + looped_out * w
            exit_entropy = -(exit_logprobs.exp() * exit_logprobs).sum(dim=-1).mean()

        # ── Cognitive Subsystems (parallel) ──
        mem_out = self.memory(core_output)
        emo_out = self.emotional(core_output)
        cre_out = self.creative(core_output)
        con_out = self.consciousness(core_output)
        hmem_out = self.hierarchical_memory(core_output)

        # ── Uncertainty ──
        uncertainty = self.uncertainty(core_output)

        # ── NeuralBus ──
        bus_flat = self.neural_bus([mem_out, emo_out, cre_out, con_out, hmem_out]) if self.config.use_neural_bus else \
                   (mem_out + emo_out + cre_out + con_out + hmem_out).mean(dim=1) / 5
        bus_out = bus_flat.unsqueeze(1).expand(-1, core_output.size(1), -1)

        # ── DeepSearch ──
        if self.config.use_deepsearch and not self.training:
            ds_out = self.deepsearch(core_output)
        else:
            ds_out = core_output

        # ── Meta-Fusion ──
        fused = self.meta_fusion(ds_out, mem_out, emo_out, cre_out, con_out)
        fused = fused + bus_out

        # ── Top Model (Cognitive Executive) ──
        reward_loss = torch.tensor(0.0, device=x.device)
        if self.top_model is not None:
            subsystem_outputs = [mem_out, emo_out, cre_out, con_out, hmem_out]

            reward_weights = None
            if self.reward_system is not None:
                reward_result = self.reward_system(
                    subsystem_outputs, uncertainty["uncertainty"]
                )
                reward_weights = reward_result["reward_weights"]
                reward_loss = reward_result["reward_loss"]

            h = self.top_model(
                fused, subsystem_outputs, uncertainty["uncertainty"],
                reward_weights=reward_weights,
            )
        else:
            h = fused

        x = self.norm(h)
        logits = self.out_up(self.out_down(x))

        # ── Loss ──
        loss = None
        if labels is not None:
            text_len = input_ids.size(1) if input_ids is not None else 0
            text_logits = logits[:, :text_len-1] if text_len > 1 else logits[:, :0]
            text_labels = labels[:, 1:] if labels.size(1) > 1 else labels[:, :0]
            if text_logits.size(1) > 0:
                loss = F.cross_entropy(
                    text_logits.reshape(-1, logits.size(-1)),
                    text_labels.reshape(-1), ignore_index=-100,
                )
                loss = loss + self.config.moe_aux_loss_weight * aux_loss
                loss = loss + self.config.mod_entropy_weight * self._mod_entropy(mod_probs_all)
                loss = loss + self.config.exit_entropy_weight * exit_entropy
                loss = loss + self.config.reward_loss_weight * reward_loss

        return {
            "logits": logits, "loss": loss, "aux_loss": aux_loss,
            "uncertainty": uncertainty["uncertainty"],
            "mod_probs": mod_probs_all,
            "reward_loss": reward_loss,
        }

    def _mod_entropy(self, mod_probs_all):
        if not mod_probs_all:
            return torch.tensor(0.0)
        s = torch.stack([p.mean(dim=0) for p in mod_probs_all], dim=0)
        return -(s * torch.log(s + 1e-8) + (1 - s) * torch.log(1 - s + 1e-8)).mean()

    @torch.no_grad()
    def generate(self, input_ids, images=None, audio=None,
                 max_new_tokens=100, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]
            logits = self(input_ids=idx, images=images, audio=audio)["logits"][:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == 1:
                break
        return input_ids

    def count_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  NICTO MASTER Parameters: {n:,} ({n/1e9:.2f}B)")
        return n


# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    configs = [
        ("TINY (validation)", config_master_tiny()),
        ("100M", config_master_100m()),
        ("1B", config_master_1b()),
        ("7B", config_master_7b()),
    ]

    for name, cfg in configs:
        print(f"\n{'='*50}")
        print(f"NICTO MASTER {name}")
        print(f"{'='*50}")
        model = NICTOMasterModel(cfg)

        # Text-only
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        out = model(input_ids=x, labels=x)
        print(f"  Text-only loss: {out['loss'].item():.4f}")

        # Text+Image
        if hasattr(cfg, 'image_size') and cfg.image_size <= 64:
            img = torch.randn(2, 3, cfg.image_size, cfg.image_size)
            out = model(input_ids=x, images=img, labels=x)
            print(f"  Text+Image loss: {out['loss'].item():.4f}")

        # Text+Audio
        if hasattr(cfg, 'audio_mel_bins'):
            aud = torch.randn(2, cfg.audio_mel_bins, 16)
            out = model(input_ids=x, audio=aud, labels=x)
            print(f"  Text+Audio loss: {out['loss'].item():.4f}")

        # Generate
        gen = model.generate(torch.randint(0, cfg.vocab_size, (1, 5)), max_new_tokens=10)
        print(f"  Generated: {gen.shape}")

        del model

    print(f"\n{'='*50}")
    print("NICTO MASTER — validated!")
    print(f"{'='*50}")
