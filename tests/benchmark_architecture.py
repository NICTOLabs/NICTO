"""
NICTO AI vs GPT-2 Architecture Benchmark
Lightweight comparison that doesn't require loading full models
"""

import torch
import time
import sys
import json
import os
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nicto_ai.core.model import NICTOModel


def get_system_info():
    """Get system information"""
    mem = psutil.virtual_memory()
    return {
        "total_ram_gb": mem.total / 1024**3,
        "available_ram_gb": mem.available / 1024**3,
        "cpu_count": psutil.cpu_count(),
    }


def measure_nicto():
    """Measure NICTO model properties"""
    print("Creating NICTO model (small config)...")
    
    model = NICTOModel(
        vocab_size=256,
        dim=128,
        max_seq_len=128,
    )
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    # Inference speed test
    input_ids = torch.randint(0, 256, (1, 32))
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)
    elapsed = time.time() - start
    
    tokens_per_sec = (32 * 5) / elapsed
    
    # Forward pass to get output shape
    with torch.no_grad():
        output = model(input_ids)
    
    logits_shape = output['logits'].shape if isinstance(output, dict) else output.logits.shape
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "tokens_per_sec": tokens_per_sec,
        "logits_shape": list(logits_shape),
        "vocab_size": 256,
        "dim": 128,
        "max_seq_len": 128,
    }


def get_gpt2_architecture():
    """Get GPT-2 architecture stats (no model loading needed)"""
    # GPT-2 standard architecture stats from public knowledge
    return {
        "total_params": 124439808,  # 124M
        "trainable_params": 124439808,
        "model_size_mb": 474.0,  # ~474 MB in FP32
        "tokens_per_sec": None,  # Cannot measure without loading
        "logits_shape": [None, None, 50257],
        "vocab_size": 50257,
        "dim": 768,
        "max_seq_len": 1024,
        "layers": 12,
        "attention_heads": 12,
        "hidden_size": 3072,
        "source": "OpenAI GPT-2 public specifications",
    }


def compare_architectures():
    """Compare NICTO vs GPT-2 architectures"""
    print("=" * 70)
    print("NICTO AI vs GPT-2 ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    system = get_system_info()
    print(f"\nSystem: {system['cpu_count']} CPUs, {system['available_ram_gb']:.1f} GB available RAM")
    
    # NICTO measurements
    print("\n--- NICTO AI (Small Config) ---")
    nicto = measure_nicto()
    
    # GPT-2 specs (from public knowledge, no loading needed)
    print("\n--- GPT-2 (Standard Config) ---")
    gpt2 = get_gpt2_architecture()
    print(f"  Parameters: {gpt2['total_params']:,}")
    print(f"  Model size: {gpt2['model_size_mb']:.1f} MB")
    print(f"  Source: Public specifications")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    param_ratio = gpt2['total_params'] / nicto['total_params']
    size_ratio = gpt2['model_size_mb'] / nicto['model_size_mb']
    
    print(f"\n{'Metric':<30} {'NICTO (small)':<20} {'GPT-2':<20} {'Ratio':<15}")
    print("-" * 85)
    print(f"{'Parameters':<30} {nicto['total_params']:>18,} {gpt2['total_params']:>18,} {param_ratio:>12.1f}x")
    print(f"{'Model Size (MB)':<30} {nicto['model_size_mb']:>17.2f} {gpt2['model_size_mb']:>17.1f} {size_ratio:>12.1f}x")
    print(f"{'Vocab Size':<30} {nicto['vocab_size']:>18,} {gpt2['vocab_size']:>18,} {gpt2['vocab_size']/nicto['vocab_size']:>12.1f}x")
    print(f"{'Hidden Dim':<30} {nicto['dim']:>18} {gpt2['dim']:>18} {gpt2['dim']/nicto['dim']:>12.1f}x")
    print(f"{'Max Seq Len':<30} {nicto['max_seq_len']:>18} {gpt2['max_seq_len']:>18} {gpt2['max_seq_len']/nicto['max_seq_len']:>12.1f}x")
    print(f"{'Inference (tok/s)':<30} {nicto['tokens_per_sec']:>17.1f} {'N/A (not loaded)':<20} {'N/A':<15}")
    
    # NICTO unique features
    print("\n" + "=" * 70)
    print("NICTO UNIQUE FEATURES (Not in GPT-2)")
    print("=" * 70)
    print("""
    1. Multi-Latent Attention (MLA)
       - KV cache compression via learned vectors
       - More memory-efficient than standard attention
    
    2. Mixture of Experts (MoE)
       - Auxiliary-loss-free load balancing
       - Only 2 experts active per token (sparse)
    
    3. Mamba (State Space Model)
       - O(N) complexity vs O(N^2) for attention
       - Selective state space model (S6)
    
    4. Liquid Neural Networks
       - Continuous-time dynamics (ODE-based)
       - Adaptive time constants
    
    5. Consciousness Layer
       - Metacognition and self-monitoring
       - Uncertainty estimation (MC Dropout)
       - Error detection
    
    6. Emotion System
       - Multi-modal emotion processing
       - Empathy generation
    
    7. Hierarchical Memory
       - Working, episodic, semantic memory
       - Memory consolidation
    
    8. DeepSearch
       - Chain-of-thought reasoning
       - Multi-step search
    
    9. Anti-Hallucination Engine
       - Claim extraction and verification
       - Grounding and source attribution
    
    10. Data Sorting
        - Token sorting for memory consolidation
        - Network priority gating
    """)
    
    # GPT-2 strengths
    print("=" * 70)
    print("GPT-2 STRENGTHS")
    print("=" * 70)
    print("""
    1. Battle-tested architecture
       - Trained on 40GB of internet text
       - Well-understood failure modes
    
    2. Optimized inference
       - Widely deployed in production
       - Extensive optimization ecosystem
    
    3. Large vocabulary (50,257 tokens)
       - Better text encoding
       - More efficient tokenization
    
    4. 1024 token context window
       - Longer text understanding
       - Better coherence
    
    5. Proven benchmark results
       - Perplexity: ~29.41 on WikiText-103
       - Well-established baseline
    """)
    
    # Full-scale projection
    print("=" * 70)
    print("FULL-SCALE NICTO PROJECTIONS")
    print("=" * 70)
    print("""
    Based on README claims (150B parameters):
    
    Metric                    NICTO (Projected)    GPT-2         Advantage
    -----------------------   -----------------    -----         ---------
    Parameters                150B                 124M          1210x
    Active per token          20B                  124M          N/A (sparse)
    Vocab Size                128,000              50,257        2.5x
    Max Context               10M tokens           1,024         9766x
    Training                  Not yet              40GB text     -
    Benchmarks                Not yet              Established   -
    
    ⚠️  NOTE: Full-scale NICTO has NOT been trained or benchmarked.
    These are architectural projections only.
    """)
    
    # Save results
    results = {
        "system": system,
        "nicto_small": nicto,
        "gpt2_standard": gpt2,
        "comparison": {
            "param_ratio": param_ratio,
            "size_ratio": size_ratio,
        },
        "nicto_unique_features": [
            "Multi-Latent Attention (MLA)",
            "Mixture of Experts (MoE)",
            "Mamba (State Space Model)",
            "Liquid Neural Networks",
            "Consciousness Layer",
            "Emotion System",
            "Hierarchical Memory",
            "DeepSearch",
            "Anti-Hallucination Engine",
            "Data Sorting",
        ],
        "disclaimer": "NICTO small config used for measurement. Full-scale (150B) has NOT been trained or benchmarked.",
    }
    
    output_file = Path(__file__).parent / "benchmark_architecture.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = compare_architectures()
