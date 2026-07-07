"""
NICTO AI - Fast Engines

Provides optimized C++ and Python implementations for:
- Byte-level tokenizer (encode/decode)
- Token sampler (top-k + softmax + multinomial)
- Autoregressive generation (pre-allocated buffer)

The C++ engine is loaded if available (10-50x faster).
Otherwise, the Python fallback is used.

To build the C++ engine:
    python -m nicto_ai.engine.build_cpp
"""

_cpp_available = False
_cpp_module = None

# Try to import C++ engine
try:
    from nicto_ai.engine import nicto_engine as _cpp_module
    _cpp_available = True
except ImportError:
    pass

# Import Python fallbacks
from nicto_ai.engine.fast_ops import (
    fast_encode,
    fast_decode,
    fast_encode_tensor,
    fast_decode_tensor,
    find_stop_token,
    fused_sample,
    fused_topk_routing,
    fused_moe_forward,
    fast_generate,
    vectorized_selective_scan,
)


def has_cpp_engine() -> bool:
    """Check if C++ engine is loaded."""
    return _cpp_available


# ============================================================
# Unified API: use C++ if available, else Python
# ============================================================

def encode(text, vocab_size=32000):
    """UTF-8 text -> token IDs."""
    if _cpp_available:
        return _cpp_module.encode(text, vocab_size)
    return fast_encode(text, vocab_size)


def decode(tokens):
    """Token IDs -> UTF-8 text."""
    if _cpp_available:
        if isinstance(tokens, list):
            return _cpp_module.decode(tokens)
        return _cpp_module.decode(tokens.tolist())
    return fast_decode(tokens)


def sample(logits, temperature=0.8, top_k=50):
    """Fused top-k + softmax + multinomial sample."""
    if _cpp_available:
        import torch
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().tolist()
        if isinstance(logits, list):
            return _cpp_module.sample(logits, len(logits), temperature, top_k)
        return _cpp_module.sample(list(logits), len(logits), temperature, top_k)
    return fused_sample(logits, temperature, top_k)


def find_stop(text):
    """Find stop token position. Returns -1 if not found."""
    if _cpp_available:
        return _cpp_module.find_stop(text)
    return find_stop_token(text)


__all__ = [
    "has_cpp_engine",
    "encode",
    "decode",
    "sample",
    "find_stop",
    "fast_encode",
    "fast_decode",
    "fast_encode_tensor",
    "fast_decode_tensor",
    "find_stop_token",
    "fused_sample",
    "fused_topk_routing",
    "fused_moe_forward",
    "fast_generate",
    "vectorized_selective_scan",
]
