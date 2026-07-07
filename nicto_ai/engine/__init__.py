"""
NICTO AI - Fast Engines

Provides optimized C++ and Python implementations for:
- Byte-level tokenizer (encode/decode)
- Mixture of Experts (fused routing + expert forward)
- Autoregressive generation (pre-allocated buffer + fused sampling)
- SSM/Liquid parallel scan

Usage:
    from nicto_ai.engine import fast_encode, fast_decode, fused_moe_forward, fast_generate

The C++ engine is loaded if available (faster). Otherwise, the Python
fallback is used (still significantly faster than the original code).
"""

import os

# Try to load C++ engine
_cpp_available = False
try:
    from nicto_ai.engine.nicto_engine import (
        fast_encode as cpp_fast_encode,
        fast_decode as cpp_fast_decode,
        fast_batch_encode,
        find_stop_token as cpp_find_stop_token,
        fused_topk as cpp_fused_topk,
        compute_load_balance_loss as cpp_compute_load_balance_loss,
        fused_sample as cpp_fused_sample,
        selective_scan as cpp_selective_scan,
        liquid_neuron_scan as cpp_liquid_neuron_scan,
    )
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
    fused_topk_routing,
    fused_moe_forward,
    fused_sample,
    fast_generate,
    vectorized_selective_scan,
)


def has_cpp_engine() -> bool:
    """Check if C++ engine is available."""
    return _cpp_available


# Use C++ versions if available, otherwise Python
if _cpp_available:
    encode = cpp_fast_encode
    decode = cpp_fast_decode
    sample = cpp_fused_sample
else:
    encode = fast_encode
    decode = fast_decode
    sample = fused_sample


__all__ = [
    "has_cpp_engine",
    "encode",
    "decode",
    "sample",
    "fast_encode",
    "fast_decode",
    "fast_encode_tensor",
    "fast_decode_tensor",
    "find_stop_token",
    "fused_topk_routing",
    "fused_moe_forward",
    "fused_sample",
    "fast_generate",
    "vectorized_selective_scan",
]
