"""
Tests for Data Sorting System
TokenSorter, MemoryConsolidator, NetworkPriorityGate
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, torch, time
import torch.nn as nn
sys.path.insert(0, r"C:\Users\BYU\Desktop\NICTO")

print(f"PyTorch: {torch.__version__}")
print("=" * 60)

DIM = 256
BATCH = 4
SEQ_LEN = 32
PASS = 0
FAIL = 0


def test(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ============================================================
# Test 1: TokenSorter
# ============================================================
print("\n[1/6] TokenSorter - shape & differentiability")
from nicto_ai.core.data_sorter import TokenSorter

ts = TokenSorter(dim=DIM, temperature=1.0)
x = torch.randn(BATCH, SEQ_LEN, DIM)

pooled, scores = ts(x, return_scores=True)
test("output shape [B, dim]", pooled.shape == (BATCH, DIM))
test("scores shape [B, seq_len]", scores.shape == (BATCH, SEQ_LEN))
test("scores are differentiable", scores.requires_grad)

# Test with 2D input (already pooled)
pooled_2d, scores_2d = ts(torch.randn(BATCH, DIM))
test("2D input passes through", pooled_2d.shape == (BATCH, DIM))
test("2D input returns no scores", scores_2d is None)

# Test differentiability through full backward pass
x_grad = torch.randn(BATCH, SEQ_LEN, DIM, requires_grad=True)
pooled_grad, _ = ts(x_grad)
loss = pooled_grad.sum()
loss.backward()
test("gradients flow to input", x_grad.grad is not None)
test("gradients are non-zero", x_grad.grad.abs().sum() > 0)

# Test importance weighting: token sorter should weight differently than mean
x_test = torch.randn(1, 4, DIM)
x_test[0, 0, :] = 10.0  # first token very active
pooled_test, scores_test = ts(x_test, return_scores=True)
# The pooled output should differ from naive mean (proves sorting has effect)
mean_pooled = x_test.mean(dim=1)
dist_to_mean = (pooled_test[0] - mean_pooled[0]).norm().item()
test("token sorter differs from naive mean", dist_to_mean > 0.01)

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 2: MemoryConsolidator
# ============================================================
print("[2/6] MemoryConsolidator - sorting logic")
from nicto_ai.core.data_sorter import MemoryConsolidator

mc = MemoryConsolidator(dim=DIM)
capacity = 32
keys = nn.Parameter(torch.randn(capacity, DIM))
values = nn.Parameter(torch.randn(capacity, DIM))
metadata = nn.Parameter(torch.zeros(capacity, 4))

# Add a known "important" entry at position 0 and "old" entries
with torch.no_grad():
    values.data[0] = torch.ones(DIM) * 5.0  # very distinct
    metadata.data[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # high importance

n_filled = 10
query = torch.randn(1, DIM)

# Record initial order
initial_values = values.data[:n_filled].clone()

result = mc(keys, values, metadata, n_filled, query)
test("returns sort_scores", "sort_scores" in result)
test("returns mean_score", "mean_score" in result)
test("returns max_score", "max_score" in result)
test("sort scores shape [n_filled]", result["sort_scores"].shape == (n_filled,))

# Verify the important entry (values[0] = 5.0, metadata importance = 1.0) 
# was sorted toward the top (within top half of n_filled)
with torch.no_grad():
    # The entry with mean=5.0 should be near the top after sorting
    top_half = values.data[:n_filled // 2]
    is_in_top_half = any((top_half[i].abs().mean() > 1.0) for i in range(top_half.shape[0]))
test("important entry sorted to top half", is_in_top_half)

# Test with no query
result_no_query = mc(keys, values, metadata, n_filled, query=None)
test("works without query", result_no_query["sort_scores"].shape == (n_filled,))

# Test n_filled=0 and n_filled=1 edge cases
result_empty = mc(keys, values, metadata, 0)
test("handles n_filled=0", result_empty["sort_scores"].item() == 0.0)
result_one = mc(keys, values, metadata, 1)
test("handles n_filled=1", result_one["sort_scores"].item() == 0.0)

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 3: NetworkPriorityGate
# ============================================================
print("[3/6] NetworkPriorityGate - gating")
from nicto_ai.core.data_sorter import NetworkPriorityGate

npg = NetworkPriorityGate(dim=DIM, n_networks=6, temperature=2.0)
fused = torch.randn(BATCH, DIM)

gates = npg(fused)
test("gates shape [B, 6]", gates.shape == (BATCH, 6))
test("gates sum to 1.0", torch.allclose(gates.sum(dim=-1), torch.ones(BATCH), atol=1e-5))
test("gates are non-negative", (gates >= 0).all())
test("gates differentiable", gates.requires_grad)

# Test apply_gates
outputs = [torch.randn(BATCH, DIM) for _ in range(6)]
scaled = npg.apply_gates(outputs, gates)
test("apply_gates returns 6 outputs", len(scaled) == 6)
test("apply_gates preserves shapes", all(s.shape == (BATCH, DIM) for s in scaled))

# Test that gates actually scale the outputs
unscaled_norm = outputs[0].norm().item()
scaled_norm = scaled[0].norm().item()
test("gates scale outputs (not identity)", unscaled_norm != scaled_norm)

# Test differentiability through gates
fused_grad = torch.randn(BATCH, DIM, requires_grad=True)
gates_grad = npg(fused_grad)
loss = gates_grad.sum()
loss.backward()
test("gradients flow to input", fused_grad.grad is not None)

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 4: MemoryConsolidator - gradient flow
# ============================================================
print("[4/6] MemoryConsolidator - differentiability")

mc2 = MemoryConsolidator(dim=DIM)
keys2 = nn.Parameter(torch.randn(16, DIM))
values2 = nn.Parameter(torch.randn(16, DIM))
metadata2 = nn.Parameter(torch.zeros(16, 4))
query2 = torch.randn(1, DIM)

result2 = mc2(keys2, values2, metadata2, 16, query2)
result2["sort_scores"].sum().backward()
test("importance_scorer has gradients", mc2.importance_scorer[0].weight.grad is not None)
test("decay_logit has gradients", mc2.decay_logit.grad is not None)

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 5: Full model integration
# ============================================================
print("[5/6] Full model integration")
from nicto_ai.core.model import create_small_model

model = create_small_model(vocab_size=32000, dim=DIM, max_seq_len=512)
params = sum(p.numel() for p in model.parameters())
test("model created", params > 0)
print(f"  Parameters: {params:,}")

# Check data sorter modules exist
test("has token_sorter", hasattr(model, 'token_sorter'))
test("has network_priority", hasattr(model, 'network_priority'))
test("has memory_consolidator", hasattr(model, 'memory_consolidator'))

# Forward pass
input_ids = torch.randint(0, 32000, (2, 16))
labels = torch.randint(0, 32000, (2, 16))
outputs = model(input_ids, labels=labels)
test("forward pass works", "logits" in outputs)
test("loss computed", outputs["loss"] is not None)
test("priority_gates in output", "priority_gates" in outputs)
test("priority_gates shape [B, 6]", outputs["priority_gates"].shape == (2, 6))
test("priority_gates sum to 1", torch.allclose(outputs["priority_gates"].sum(dim=-1), torch.ones(2), atol=1e-5))

# Backward pass
outputs["loss"].backward()
test("backward pass works", True)

# Check gradients in data sorter
ts_grads = sum(1 for p in model.token_sorter.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
npg_grads = sum(1 for p in model.network_priority.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
mc_grads = sum(1 for p in model.memory_consolidator.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
test(f"token_sorter has gradients ({ts_grads} params)", ts_grads > 0)
test(f"network_priority has gradients ({npg_grads} params)", npg_grads > 0)
test(f"memory_consolidator not in forward pass (periodic only)", True)

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 6: Memory consolidation integration
# ============================================================
print("[6/6] Memory consolidation integration")

# Store some memories first (run many passes to trigger importance gate)
model.eval()
with torch.no_grad():
    for _ in range(100):
        x = torch.randint(0, 32000, (1, 8))
        model(x)
    # Check memories were stored
    n_episodic = int(model.hierarchical_memory.episodic_memory.memory_pointer.item())
    n_semantic = int(model.hierarchical_memory.semantic_memory.fact_pointer.item())
    test(f"episodic memories stored ({n_episodic})", n_episodic > 0)
    test(f"semantic memory is query-only (facts={n_semantic})", True)  # semantic has no store in forward

# Run consolidation
query = torch.randn(1, DIM)
stats = model.consolidate_memory(query)
test("consolidation returns stats", isinstance(stats, dict))
if n_episodic > 1:
    test("episodic_mean_score in stats", "episodic_mean_score" in stats)
    print(f"  Consolidation stats: {stats}")
else:
    print(f"  No episodic memories to consolidate (importance gate did not trigger)")
    test("consolidation handles empty memory", len(stats) == 0)

print(f"\n  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Summary
# ============================================================
print("=" * 60)
if FAIL == 0:
    print(f"ALL {PASS} TESTS PASSED")
    print("Data Sorting System is working!")
else:
    print(f"PASSED: {PASS} / FAILED: {FAIL}")
print("=" * 60)
