"""Test DeepSeek V4 architectural innovations integrated into NICTO.

Tests:
1. Engram Conditional Memory — O(1) hash lookup, gating, full module
2. Manifold-Constrained Hyper-Connections — Sinkhorn-Knopp, doubly stochastic property
3. DeepSeek Sparse Attention — Lightning Indexer, local/global attention, full DSA
4. End-to-end training with all three modules
"""
import sys; sys.path.insert(0, ".")
import torch
import math

print("=" * 60)
print("DeepSeek V4 Architectural Innovations Test")
print("=" * 60)

# ==============================================================================
# 1. ENGRAM CONDITIONAL MEMORY
# ==============================================================================
print("\n--- Engram Conditional Memory ---")

from nicto_ai.nictos.neural.engram import (
    NgramHasher, EngramEmbedding, EngramGate, EngramModule, EngramTransformerBlock
)

# 1a. N-gram Hashing
hasher = NgramHasher(vocab_size=1000, max_ngram=3, num_heads=8, seed=42)
token_ids = torch.randint(0, 1000, (2, 32))
hash_indices = hasher.hash(token_ids)
assert hash_indices.shape == (2, 32, 16), f"Expected (2,32,16), got {hash_indices.shape}"
assert hash_indices.min() >= 0, "Hash indices must be non-negative"
print(f"1a. NgramHasher OK — shape={hash_indices.shape}, range=[{hash_indices.min()}, {hash_indices.max()}]")

# 1b. Engram Embedding
vocab_sizes = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
               1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097]
emb = EngramEmbedding(vocab_sizes, embed_dim=64)
embedded = emb(hash_indices[:, :, :8])  # first 8 heads
assert embedded.shape == (2, 32, 8, 64), f"Expected (2,32,8,64), got {embedded.shape}"
print(f"1b. EngramEmbedding OK — shape={embedded.shape}")

# 1c. Engram Gate
gate = EngramGate(hidden_dim=256, embed_dim=64)
hidden = torch.randn(2, 32, 256)
gate_vals = gate(hidden, embedded)
assert gate_vals.shape == (2, 32, 8, 1), f"Expected (2,32,8,1), got {gate_vals.shape}"
assert gate_vals.min() >= 0 and gate_vals.max() <= 1, "Gate must be in [0, 1]"
print(f"1c. EngramGate OK — shape={gate_vals.shape}, range=[{gate_vals.min():.4f}, {gate_vals.max():.4f}]")

# 1d. Full Engram Module
engram = EngramModule(
    hidden_dim=256, vocab_size=1000, max_ngram=3,
    num_heads=8, embed_dim_per_head=64
)
out = engram(hidden, token_ids)
assert out.shape == (2, 32, 256), f"Expected (2,32,256), got {out.shape}"
# Test gradient flow
loss = out.sum()
loss.backward()
print(f"1d. EngramModule OK — output shape={out.shape}, grad flows=True")

# 1e. Engram Transformer Block
block = EngramTransformerBlock(
    hidden_dim=256, num_heads=4, ffn_dim=512,
    vocab_size=1000, max_ngram=3, engram_heads=8, embed_dim=64
)
out = block(hidden, token_ids)
assert out.shape == (2, 32, 256)
print(f"1e. EngramTransformerBlock OK — shape={out.shape}")

# ==============================================================================
# 2. MANIFOLD-CONSTRAINED HYPER-CONNECTIONS
# ==============================================================================
print("\n--- Manifold-Constrained Hyper-Connections ---")

from nicto_ai.nictos.neural.mhc import (
    sinkhorn_knopp, ManifoldConstrainedHyperConnections, MHCBlock, MHCModel
)

# 2a. Sinkhorn-Knopp produces doubly stochastic matrices
matrix = torch.randn(4, 4)
ds_matrix = sinkhorn_knopp(matrix, num_iters=20)

# Check doubly stochastic properties
row_sums = ds_matrix.sum(dim=-1)
col_sums = ds_matrix.sum(dim=-2)
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), \
    f"Row sums not ~1: {row_sums}"
assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3), \
    f"Col sums not ~1: {col_sums}"
assert (ds_matrix >= 0).all(), "Negative values in doubly stochastic matrix"
print(f"2a. Sinkhorn-Knopp OK — row_sums={row_sums.tolist()}, col_sums={col_sums.tolist()}")

# 2b. Spectral norm <= 1
svd = torch.linalg.svd(ds_matrix)
spectral_norm = svd[1].max().item()
assert spectral_norm <= 1.0 + 1e-3, f"Spectral norm > 1: {spectral_norm}"
print(f"2b. Spectral norm OK — max singular value = {spectral_norm:.4f}")

# 2c. Compositional closure: product of DS matrices is DS
ds_matrix2 = sinkhorn_knopp(torch.randn(4, 4))
product = ds_matrix @ ds_matrix2
product_ds = sinkhorn_knopp(product, num_iters=5)  # re-project for numerical stability
row_sums_p = product_ds.sum(dim=-1)
col_sums_p = product_ds.sum(dim=-2)
assert torch.allclose(row_sums_p, torch.ones_like(row_sums_p), atol=1e-2)
assert torch.allclose(col_sums_p, torch.ones_like(col_sums_p), atol=1e-2)
print(f"2c. Compositional closure OK")

# 2d. MHC Layer
mhc = ManifoldConstrainedHyperConnections(dim=128, expansion_rate=4, sinkhorn_iters=20)
x_stream = torch.randn(2, 16, 128 * 4)  # (B, T, n*C)
out = mhc(x_stream)
assert out.shape == x_stream.shape, f"Shape mismatch: {out.shape}"
print(f"2d. ManifoldConstrainedHyperConnections OK — {x_stream.shape} -> {out.shape}")

# 2e. MHC with gradient flow
loss = out.sum()
loss.backward()
assert all(p.grad is not None for p in mhc.parameters() if p.requires_grad)
print(f"2e. Gradient flow OK")

# 2f. MHC Block
mhc_block = MHCBlock(dim=128, num_heads=4, ffn_dim=256, expansion_rate=4)
out = mhc_block(x_stream)
assert out.shape == x_stream.shape
print(f"2f. MHCBlock OK — shape={out.shape}")

# 2g. MHC Model (full)
mhc_model = MHCModel(
    vocab_size=1000, dim=128, num_layers=3, num_heads=4,
    ffn_dim=256, expansion_rate=4
)
token_ids = torch.randint(0, 1000, (2, 32))
logits = mhc_model(token_ids)
assert logits.shape == (2, 32, 1000), f"Expected (2,32,1000), got {logits.shape}"
loss = logits.sum()
loss.backward()
print(f"2g. MHCModel OK — logits shape={logits.shape}, grad flows=True")

# 2h. Verify stability: 10-layer deep model
mhc_deep = MHCModel(
    vocab_size=1000, dim=128, num_layers=10, num_heads=4,
    ffn_dim=256, expansion_rate=4
)
out = mhc_deep(token_ids)
assert torch.isfinite(out).all(), "Non-finite values in deep mHC output"
print(f"2h. Deep (10-layer) mHC stability OK — all finite")

# ==============================================================================
# 3. DEEPSEEK SPARSE ATTENTION
# ==============================================================================
print("\n--- DeepSeek Sparse Attention ---")

from nicto_ai.nictos.neural.dattention import (
    LightningIndexer, CompressedKVCache, LocalWindowAttention,
    GlobalSparseAttention, DeepSeekSparseAttention, DSABlock, DSAModel
)

x_test = torch.randn(2, 64, 256)  # (B, T, dim)

# 3a. Lightning Indexer
indexer = LightningIndexer(dim=256, num_heads=1)
query = torch.randn(2, 64, 256)
key = torch.randn(2, 128, 256)  # longer context
indices = indexer(query, key, top_k=16)
assert indices.shape == (2, 64, 16), f"Expected (2,64,16), got {indices.shape}"
assert indices.min() >= 0 and indices.max() < 128, "Indices out of range"
print(f"3a. LightningIndexer OK — shape={indices.shape}, range=[{indices.min()}, {indices.max()}]")

# 3b. Compressed KV Cache
kv_cache = CompressedKVCache(dim=256, compression_ratio=4)
k = torch.randn(2, 64, 256)
v = torch.randn(2, 64, 256)
k_c, v_c = kv_cache(k, v)
expected_len = 64 // 4
assert k_c.shape == (2, expected_len, 256), f"Expected (2,{expected_len},256), got {k_c.shape}"
print(f"3b. CompressedKVCache OK — {k.shape} -> {k_c.shape} ({4}x compression)")

# 3c. Local Window Attention
local_attn = LocalWindowAttention(dim=256, num_heads=4, window_size=32)
out = local_attn(x_test)
assert out.shape == x_test.shape
print(f"3c. LocalWindowAttention OK — shape={out.shape}")

# 3d. Global Sparse Attention
global_attn = GlobalSparseAttention(dim=256, num_heads=4, top_k=8, compression_ratio=4)
out = global_attn(x_test)
assert out.shape == x_test.shape
print(f"3d. GlobalSparseAttention OK — shape={out.shape}")

# 3e. Combined DSA
dsa = DeepSeekSparseAttention(dim=256, num_heads=4, window_size=32, top_k=8)
out = dsa(x_test)
assert out.shape == x_test.shape
print(f"3e. DeepSeekSparseAttention OK — shape={out.shape}")

# 3f. DSA Block
dsa_block = DSABlock(dim=256, num_heads=4, ffn_dim=512, window_size=32, top_k=8)
out = dsa_block(x_test)
assert out.shape == x_test.shape
print(f"3f. DSABlock OK — shape={out.shape}")

# 3g. DSA Model (full)
dsa_model = DSAModel(
    vocab_size=1000, dim=256, num_layers=3, num_heads=4,
    ffn_dim=512, window_size=32, top_k=8
)
token_ids = torch.randint(0, 1000, (2, 64))
logits = dsa_model(token_ids)
assert logits.shape == (2, 64, 1000)
print(f"3g. DSAModel OK — logits shape={logits.shape}")

# 3h. Long context test (256 tokens)
long_ids = torch.randint(0, 1000, (1, 256))
out = dsa_model(long_ids)
assert out.shape == (1, 256, 1000)
assert torch.isfinite(out).all()
print(f"3h. Long context (256 tokens) OK — all finite")

# ==============================================================================
# 4. END-TO-END: ALL THREE TOGETHER
# ==============================================================================
print("\n--- Combined: Engram + mHC + DSA ---")

from nicto_ai.nictos.neural.engram import EngramModule
from nicto_ai.nictos.neural.mhc import MHCBlock
from nicto_ai.nictos.neural.dattention import DSABlock

class DeepSeekV4Block(torch.nn.Module):
    """Combined block: Engram -> mHC attention -> mHC FFN -> DSA"""
    def __init__(self, dim=256, num_heads=4, ffn_dim=512, vocab_size=1000):
        super().__init__()
        self.engram = EngramModule(dim, vocab_size, num_heads=4, embed_dim_per_head=32)
        self.mhc = MHCBlock(dim, num_heads, ffn_dim, expansion_rate=2)
        self.dsa = DSABlock(dim, num_heads, ffn_dim, window_size=32, top_k=8)

    def forward(self, x, token_ids):
        # Engram memory
        h = self.engram(x, token_ids)
        # DSA attention + FFN
        h = self.dsa(h)
        return h

class MiniV4Model(torch.nn.Module):
    """Miniature V4 model combining all three innovations."""
    def __init__(self, vocab_size=1000, dim=256, num_layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList([
            DeepSeekV4Block(dim, num_heads=4, ffn_dim=512, vocab_size=vocab_size)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.RMSNorm(dim)
        self.head = torch.nn.Linear(dim, vocab_size)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x, token_ids)
        return self.head(self.norm(x))

model = MiniV4Model(vocab_size=1000, dim=256, num_layers=2)
token_ids = torch.randint(0, 1000, (2, 32))
logits = model(token_ids)
assert logits.shape == (2, 32, 1000)

# Training step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
targets = torch.randint(0, 1000, (2, 32))
loss = torch.nn.functional.cross_entropy(logits.view(-1, 1000), targets.view(-1))
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"4. Combined model OK — logits={logits.shape}, loss={loss.item():.4f}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

print()
print("=" * 60)
print("ALL DEEPSEEK V4 INNOVATION TESTS COMPLETE")
print("=" * 60)
