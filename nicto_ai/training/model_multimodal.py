"""
NICTO Multimodal — Extends NICTO Unified to handle vision and audio like Gemini
=================================================================================
Adds:
  1. VisionEncoder: ViT-style patch embedding + encoder for images
  2. AudioEncoder: Mel-spectrogram CNN encoder for speech/audio
  3. MultimodalProjector: Projects vision/audio features to model dim
  4. NICTOMultimodalModel: Accepts text + images + audio interleaved

Architecture flow:
  Text tokens  → Embedding
  Image        → Patch Embed → ViT Encoder → Projector → Visual tokens
  Audio        → Mel-Spec    → CNN Encoder  → Projector → Audio tokens
    → Concatenate all modality tokens
    → NOVA Core blocks (with cross-attention for visual tokens)
    → NICTO Subsystems
    → Meta-Fusion Gate
    → Output Head (text-only for now)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from nicto_ai.training.model_unified import (
    NICTOUnifiedModel, NICTOUnifiedConfig,
    UnifiedNOVABlock, MemorySubsystem, EmotionalSubsystem,
    CreativeSubsystem, ConsciousnessSubsystem, MetaFusionGate,
    RMSNorm, RotaryEmbedding, apply_rope,
)


# ============================================================
# MULTIMODAL CONFIG
# ============================================================

@dataclass
class NICTOMultimodalConfig(NICTOUnifiedConfig):
    # Vision
    image_size: int = 224
    patch_size: int = 16
    vision_dim: int = 768
    vision_layers: int = 6
    vision_heads: int = 8
    # Audio
    audio_mel_bins: int = 80
    audio_max_frames: int = 1024
    audio_dim: int = 256
    audio_layers: int = 4
    audio_conv_kernel: int = 3
    # Multimodal
    vision_token: int = 32000  # special token marking visual tokens
    audio_token: int = 32001   # special token marking audio tokens
    # Cross-attention in NOVA
    cross_attn_every: int = 2  # add cross-attn to NOVA blocks every N layers


# ============================================================
# VISION ENCODER (ViT-style)
# ============================================================

class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionEncoder(nn.Module):
    """ViT-based vision encoder that converts images to visual token sequences."""
    def __init__(self, config: NICTOMultimodalConfig):
        super().__init__()
        d = config.vision_dim
        self.patch_embed = PatchEmbed(config.image_size, config.patch_size, embed_dim=d)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d) * 0.02)
        self.pos_drop = nn.Dropout(p=0.0)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, config.vision_heads, d * 4,
                                       activation=F.gelu, batch_first=True, norm_first=True)
            for _ in range(config.vision_layers)
        ])
        self.norm = nn.LayerNorm(d, eps=config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 1:]  # remove cls token, return patch tokens


# ============================================================
# AUDIO ENCODER (Mel-spectrogram CNN)
# ============================================================

class AudioEncoder(nn.Module):
    """
    Audio encoder: mel-spectrogram → CNN stack → transformer.
    Accepts mel-spectrograms of shape (B, mel_bins, T).
    """
    def __init__(self, config: NICTOMultimodalConfig):
        super().__init__()
        d = config.audio_dim
        self.mel_bins = config.audio_mel_bins
        self.max_frames = config.audio_max_frames

        # CNN front-end
        self.conv_stack = nn.Sequential(
            nn.Conv1d(config.audio_mel_bins, d, kernel_size=config.audio_conv_kernel,
                      padding=config.audio_conv_kernel // 2),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, config.audio_max_frames // 4, d) * 0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, 4, d * 4, activation=F.gelu,
                                       batch_first=True, norm_first=True)
            for _ in range(config.audio_layers)
        ])
        self.norm = nn.LayerNorm(d, eps=config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv_stack(x)
        x = x.transpose(1, 2)
        T = x.size(1)
        x = x + self.pos_embed[:, :T]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


# ============================================================
# MULTIMODAL PROJECTOR
# ============================================================

class MultimodalProjector(nn.Module):
    """Projects vision/audio features to the model's hidden dimension."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============================================================
# CROSS-ATTENTION LAYER
# ============================================================

class CrossAttention(nn.Module):
    """Cross-attends over visual/audio tokens from the NOVA hidden state."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            attn = attn + context_mask.unsqueeze(1).unsqueeze(1)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


# ============================================================
# NICTO MULTIMODAL MODEL
# ============================================================

class NICTOMultimodalModel(NICTOUnifiedModel):
    """
    NICTO Multimodal: text + images + audio in a single unified model.
    Like Gemini — all modalities processed by the same backbone.
    """
    def __init__(self, config: NICTOMultimodalConfig):
        # We need to init the parent's modules manually to avoid double embedding
        nn.Module.__init__(self)
        self.config = config
        d = config.dim

        # Text embedding
        self.tok_emb = nn.Embedding(config.vocab_size + 2, d)  # +2 for vision/audio tokens
        self.rope = RotaryEmbedding(d // config.n_heads, config.max_seq_len, config.rope_theta)

        # Vision
        self.vision_encoder = VisionEncoder(config)
        self.vision_projector = MultimodalProjector(config.vision_dim, d)

        # Audio
        self.audio_encoder = AudioEncoder(config)
        self.audio_projector = MultimodalProjector(config.audio_dim, d)

        # NOVA core blocks (enhanced with cross-attention)
        self.nova_blocks = nn.ModuleList()
        for i in range(config.n_layers):
            block = UnifiedNOVABlock(config, layer_idx=i)
            self.nova_blocks.append(block)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d, config.n_heads) if (i % config.cross_attn_every == 0) else nn.Identity()
            for i in range(config.n_layers)
        ])

        # NICTO subsystems
        self.memory = MemorySubsystem(d, config.n_heads, config.memory_layers)
        self.emotional = EmotionalSubsystem(d, config.n_heads, config.emotional_layers, config.max_seq_len)
        self.creative = CreativeSubsystem(d, config.n_heads, config.creative_layers, config.max_seq_len)
        self.consciousness = ConsciousnessSubsystem(d, config.consciousness_dim)

        # Meta-fusion
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
        elif isinstance(module, nn.Embedding) or isinstance(module, nn.Parameter):
            pass
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        input_ids: (B, L) text tokens
        images:    (B, 3, H, W) image tensors
        audio:     (B, mel_bins, T) mel-spectrograms
        labels:    (B, L) text labels (only computed on text positions)
        """
        device = next(self.parameters()).device
        all_tokens = []
        modality_mask = []  # 0=text, 1=vision, 2=audio

        # --- Process text ---
        if input_ids is not None:
            B, L = input_ids.shape
            text_emb = self.tok_emb(input_ids)
            all_tokens.append(text_emb)
            modality_mask.append(torch.zeros(B, L, device=device))

        # --- Process images ---
        if images is not None:
            B = images.shape[0]
            vis_feats = self.vision_encoder(images)
            vis_tokens = self.vision_projector(vis_feats)
            all_tokens.append(vis_tokens)
            modality_mask.append(torch.ones(B, vis_tokens.size(1), device=device))

        # --- Process audio ---
        if audio is not None:
            B = audio.shape[0]
            aud_feats = self.audio_encoder(audio)
            aud_tokens = self.audio_projector(aud_feats)
            all_tokens.append(aud_tokens)
            modality_mask.append(torch.full((B, aud_tokens.size(1)), 2, device=device))

        if not all_tokens:
            raise ValueError("At least one modality input required")

        # --- Concatenate all tokens ---
        x = torch.cat(all_tokens, dim=1)
        modality_mask = torch.cat(modality_mask, dim=1)
        seq_len = x.size(1)

        # --- NOVA blocks with cross-attention over visual/audio tokens ---
        aux_loss = torch.tensor(0.0, device=x.device)
        prs_state = None
        mod_probs_all = []

        for i, (block, cross_attn) in enumerate(zip(self.nova_blocks, self.cross_attn_layers)):
            x, block_aux, prs_state, mod_probs = block(x, prs_state)
            aux_loss = aux_loss + block_aux
            mod_probs_all.append(mod_probs)

            # Cross-attention: query text positions over vision/audio tokens
            if isinstance(cross_attn, CrossAttention) and images is not None:
                vis_mask = (modality_mask == 1)
                text_mask = (modality_mask == 0)
                if vis_mask.any() and text_mask.any():
                    for b in range(B):
                        text_idx = text_mask[b]
                        vis_idx = vis_mask[b]
                        if text_idx.any() and vis_idx.any():
                            text_h = x[b:b+1, text_idx]
                            vis_h = x[b:b+1, vis_idx]
                            updated = cross_attn(text_h, vis_h)
                            x[b:b+1, text_idx] = updated

        core_output = x

        # --- NICTO subsystems ---
        mem_out = self.memory(core_output)
        emo_out = self.emotional(core_output)
        cre_out = self.creative(core_output)
        con_out = self.consciousness(core_output)

        # --- Meta-fusion ---
        fused = self.meta_fusion(core_output, mem_out, emo_out, cre_out, con_out)
        x = self.norm(fused)

        # --- Output (only text tokens for loss) ---
        logits = self.out_up(self.out_down(x))

        loss = None
        if labels is not None:
            text_len = input_ids.size(1) if input_ids is not None else 0
            text_logits = logits[:, :text_len - 1] if text_len > 1 else logits[:, :0]
            text_labels = labels[:, 1:] if labels.size(1) > 1 else labels[:, :0]
            if text_logits.size(1) > 0:
                loss = F.cross_entropy(
                    text_logits.reshape(-1, logits.size(-1)),
                    text_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + self.config.moe_aux_loss_weight * aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]
            logits = self(input_ids=idx, images=images, audio=audio)["logits"][:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == 1:
                break
        return input_ids

    def count_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  NICTO Multimodal Parameters: {n:,} ({n/1e9:.2f}B)")
        return n


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    configs = [
        ("Tiny (validation)", NICTOMultimodalConfig(
            vocab_size=32000, dim=128, n_heads=4, n_kv_heads=2,
            n_layers=2, max_seq_len=256, ffn_dim=256,
            ssm_d_state=4, ssm_d_conv=2, ssm_expand=2,
            attn_window_size=32, attn_n_global_tokens=4,
            moe_experts=2, moe_activated_min=1, moe_activated_max=2,
            prs_dim=32, prs_n_heads=2,
            memory_layers=1, emotional_layers=1, creative_layers=1, consciousness_dim=16,
            image_size=32, patch_size=8,  # tiny image for validation
            vision_dim=32, vision_layers=2, vision_heads=2,
            audio_mel_bins=16, audio_max_frames=32, audio_dim=16, audio_layers=1,
        )),
        ("100M", NICTOMultimodalConfig(
            vocab_size=32000, dim=768, n_heads=12, n_kv_heads=4,
            n_layers=12, max_seq_len=2048, ffn_dim=3072,
            ssm_d_state=16, ssm_d_conv=4, ssm_expand=2,
            attn_window_size=128, attn_n_global_tokens=32,
            moe_experts=4, moe_activated_min=1, moe_activated_max=2,
            prs_dim=128, prs_n_heads=2,
            memory_layers=2, emotional_layers=2, creative_layers=2, consciousness_dim=64,
        )),
    ]

    for name, cfg in configs:
        print(f"\n{'='*50}")
        print(f"NICTO Multimodal {name}")
        print(f"{'='*50}")
        model = NICTOMultimodalModel(cfg)

        # Text-only forward
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        out = model(input_ids=x, labels=x)
        print(f"  Text-only loss: {out['loss'].item():.4f}")

        # Image input
        if cfg.image_size <= 64:
            img = torch.randn(2, 3, cfg.image_size, cfg.image_size)
            out = model(input_ids=x, images=img, labels=x)
            print(f"  Text+Image loss: {out['loss'].item():.4f}")

        # Audio input
        if hasattr(cfg, 'audio_mel_bins'):
            aud = torch.randn(2, cfg.audio_mel_bins, 16)
            out = model(input_ids=x, audio=aud, labels=x)
            print(f"  Text+Audio loss: {out['loss'].item():.4f}")

        # Full multimodal
        if cfg.image_size <= 64:
            out = model(input_ids=x, images=img, audio=aud, labels=x)
            print(f"  Text+Image+Audio loss: {out['loss'].item():.4f}")

        del model

    print(f"\n{'='*50}")
    print("NICTO Multimodal validated!")
    print(f"{'='*50}")
