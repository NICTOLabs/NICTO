import sys, os, time
sys.path.insert(0, '.')
import torch

print("=" * 60)
print("  NICTO C++ ENGINE BENCHMARK")
print("=" * 60)

# Check C++ engine
from nicto_ai.engine import has_cpp_engine
print("C++ engine loaded:", has_cpp_engine())

from nicto_ai.engine import nicto_engine as cpp
from nicto_ai.engine.fast_ops import fast_encode, fast_decode, fused_sample as py_sample

# ============================================================
# 1. TOKENIZER
# ============================================================
print("\n--- 1. TOKENIZER ---")

text_short = "Hello NICTO!"
text_med = "The quick brown fox jumps over the lazy dog. " * 10
text_long = "NICTO is an experimental AI architecture. " * 100

for label, text in [("Short (12 chars)", text_short), ("Medium (450 chars)", text_med), ("Long (4200 chars)", text_long)]:
    # Python
    t0 = time.perf_counter()
    for _ in range(10000):
        tokens = fast_encode(text, 32000)
        decoded = fast_decode(tokens)
    py_time = time.perf_counter() - t0

    # C++
    t0 = time.perf_counter()
    for _ in range(10000):
        tokens = cpp.encode(text, 32000)
        decoded = cpp.decode(tokens)
    cpp_time = time.perf_counter() - t0

    speedup = py_time / cpp_time
    print(f"  {label}: Python {py_time*100:.1f}ms | C++ {cpp_time*100:.1f}ms | {speedup:.1f}x faster")

# ============================================================
# 2. SAMPLING
# ============================================================
print("\n--- 2. SAMPLING ---")

for vocab_size in [256, 32000, 128000]:
    logits_tensor = torch.tensor([0.1] * vocab_size, dtype=torch.float32)
    logits_list = [0.1] * vocab_size

    t0 = time.perf_counter()
    for _ in range(10000):
        py_sample(logits_tensor, 0.8, 50)
    py_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10000):
        cpp.sample(logits_list, vocab_size, 0.8, 50)
    cpp_time = time.perf_counter() - t0

    speedup = py_time / cpp_time
    print(f"  Vocab {vocab_size:>6}: Python {py_time*100:.1f}ms | C++ {cpp_time*100:.1f}ms | {speedup:.1f}x faster")

# ============================================================
# 3. FIND STOP
# ============================================================
print("\n--- 3. FIND STOP TOKEN ---")

text = "Hello world\n\nAssistant: Hi there" * 100

t0 = time.perf_counter()
for _ in range(10000):
    from nicto_ai.engine.fast_ops import find_stop_token
    find_stop_token(text)
py_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(10000):
    cpp.find_stop(text)
cpp_time = time.perf_counter() - t0

print(f"  Python: {py_time*100:.1f}ms | C++: {cpp_time*100:.1f}ms | {py_time/cpp_time:.1f}x faster")

print("\n" + "=" * 60)
print("  BENCHMARK COMPLETE")
print("=" * 60)
