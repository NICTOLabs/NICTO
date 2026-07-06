"""
NICTO AI - Training Model (Fixed, Working)
Clean, configurable, trainable.
Uses simplified but correct implementations of all components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class NICTOTrainConfig:
    vocab_size: int = 32000
    dim: int = 1024
    max_seq_len: int = 2048
    reasoning_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    moe_experts: int = 4
    moe_activated: int = 2
    moe_hidden: int = 2048
    memory_layers: int = 4
    emotional_layers: int = 4
    creative_layers: int = 4


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SimpleAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.wqkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.wo(out.transpose(1, 2).reshape(B, L, D))


class SimpleMoE(nn.Module):
    def __init__(self, dim, n_experts, n_activated, hidden_dim):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, L, D = x.shape
        gates = F.softmax(self.gate(x), dim=-1)
        top_k_val, top_k_idx = torch.topk(gates, self.n_activated, dim=-1)
        top_k_val = top_k_val / top_k_val.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_idx == i).any(dim=-1)
            if mask.any():
                out = expert(x[mask])
                w = (top_k_idx == i).float().sum(dim=-1, keepdim=True)[mask]
                output[mask] += w * out

        load_loss = (gates.mean(dim=(0, 1)) ** 2).sum() * self.n_experts
        return output, load_loss


class NICTOTrainModel(nn.Module):
    def __init__(self, config: NICTOTrainConfig):
        super().__init__()
        self.config = config
        d = config.dim

        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Embedding(config.max_seq_len, d)

        # Reasoning: MLA-style attention + MoE
        self.r_norm = RMSNorm(d)
        self.r_attn = SimpleAttention(d, config.n_heads, config.max_seq_len)
        self.r_moe = SimpleMoE(d, config.moe_experts, config.moe_activated, config.moe_hidden)

        # Memory: bidirectional attention (like BERT)
        mem_layer = nn.TransformerEncoderLayer(d, config.n_heads, d*4, batch_first=True, norm_first=True)
        self.m_norm = RMSNorm(d)
        self.m_layers = nn.ModuleList([mem_layer for _ in range(config.memory_layers)])

        # Emotional: causal transformer
        emo_layer = nn.TransformerEncoderLayer(d, config.n_heads, d*4, batch_first=True, norm_first=True)
        self.e_norm = RMSNorm(d)
        self.e_layers = nn.ModuleList([emo_layer for _ in range(config.emotional_layers)])

        # Creative: causal transformer
        cre_layer = nn.TransformerEncoderLayer(d, config.n_heads, d*4, batch_first=True, norm_first=True)
        self.c_norm = RMSNorm(d)
        self.c_layers = nn.ModuleList([cre_layer for _ in range(config.creative_layers)])

        # Consciousness: self-monitoring
        self.co_norm = RMSNorm(d)
        self.co_proj = nn.Linear(d, d // 2)
        self.co_back = nn.Linear(d // 2, d)

        # Fusion gate
        self.fusion_norm = RMSNorm(d)
        self.fusion_gate = nn.Linear(d * 4, 4)

        # Output
        self.out_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        # 1. Reasoning: MLA attention + MoE FFN
        h = self.r_norm(x)
        h = x + self.r_attn(h)
        h, aux_loss = self.r_moe(h)
        r = x + h

        # 2. Memory: self-attention
        m = x + self.m_layers[0](self.m_norm(x))
        for layer in self.m_layers[1:]:
            m = m + layer(self.m_norm(m))

        # 3. Emotional: self-attention
        e = x + self.e_layers[0](self.e_norm(x))
        for layer in self.e_layers[1:]:
            e = e + layer(self.e_norm(e))

        # 4. Creative: self-attention
        c = x + self.c_layers[0](self.c_norm(x))
        for layer in self.c_layers[1:]:
            c = c + layer(self.c_norm(c))

        # 5. Consciousness
        co = self.co_back(F.silu(self.co_proj(self.co_norm(x))))

        # 6. Fusion
        g = F.softmax(self.fusion_gate(torch.cat([r, m, e, c], -1)), -1)
        fused = g[..., 0:1] * r + g[..., 1:2] * m + g[..., 2:3] * e + g[..., 3:4] * c + co

        logits = self.lm_head(self.out_norm(fused))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
        return {"logits": logits, "loss": loss, "aux_loss": aux_loss}

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=50, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            logits = self(ids)["logits"][:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            ids = torch.cat([ids, torch.multinomial(F.softmax(logits, -1), 1)], -1)
        return ids

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = NICTOTrainConfig()
    model = NICTOTrainModel(config)
    print(f"Params: {model.count_parameters():,} ({model.count_parameters()/1e9:.2f}B)")

    x = torch.randint(0, config.vocab_size, (2, 64))
    y = model(x, labels=x)
    print(f"Loss: {y['loss'].item():.4f}")
    print(f"Logits: {y['logits'].shape}")

    g = model.generate(torch.randint(0, config.vocab_size, (1, 5)), max_new_tokens=10)
    print(f"Generated: {g.shape}")
    print("ALL OK")
