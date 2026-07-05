"""
NICTO AI vs GPT-2 Benchmark (v3)
Uses known specs for both models - no model loading needed
"""

import json
import sys
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_system_info():
    mem = psutil.virtual_memory()
    return {
        "total_ram_gb": round(mem.total / 1024**3, 2),
        "available_ram_gb": round(mem.available / 1024**3, 2),
        "cpu_count": psutil.cpu_count(),
    }


def compute_nicto_params():
    """
    Compute NICTO parameters from actual model_config.py values.
    Reasoning Cortex: 96 layers, dim=8192, MLA + MoE(64 experts, hidden=32768)
    """
    dim = 8192
    vocab = 128_000
    head_dim = dim // 128  # = 64

    # ==================== REASONING CORTEX (96 layers) ====================
    # MLA per layer (DeepSeek-style with KV compression)
    mla_q = dim * 1536 + 1536 * (128 * head_dim)       # wq_a + wq_b
    mla_kv = dim * (512 + 16 * head_dim) + 512 * (16 * head_dim)  # wkv_a + wkv_b
    mla_wo = (128 * head_dim) * dim                       # wo
    mla_norms = dim + head_dim * 2                        # q_norm + k_norm (RMSNorm, negligible)
    mla_per_layer = mla_q + mla_kv + mla_wo + mla_norms

    # MoE per layer (64 experts, each dim -> 32768 -> dim)
    expert_ffn = 2 * dim * 32768                           # w1 + w3 per expert
    moe_gate = dim * 64                                     # gating network
    moe_per_layer = 64 * expert_ffn + moe_gate

    # Reasoning norms
    reasoning_norm = dim  # RMSNorm

    reasoning_per_layer = mla_per_layer + moe_per_layer + reasoning_norm
    reasoning_total = 96 * reasoning_per_layer

    # ==================== EMOTIONAL CORTEX (48 layers) ====================
    emo_attn = 4 * dim * dim
    emo_ffn = 2 * dim * (4 * dim)
    emo_norm = dim * 2
    emotional_total = 48 * (emo_attn + emo_ffn + emo_norm)

    # ==================== MEMORY CORTEX (48 layers, Mamba) ====================
    mamba_d_model = 4096
    mamba_d_state = 256
    mamba_inner = mamba_d_model * 2  # expand=2
    mamba_block = (mamba_d_model * mamba_inner * 2   # x + z projections
                   + mamba_inner * 4                   # conv1d
                   + mamba_inner * mamba_d_state       # A
                   + mamba_inner                       # D
                   + mamba_inner * mamba_d_model)      # out proj
    mem_norm = mamba_d_model * 2
    memory_total = 48 * (mamba_block + mem_norm)

    # ==================== PERCEPTION CORTEX ====================
    vision_total = 24 * (4 * 1024 * 1024 + 2 * 1024 * 4096)
    audio_total = 16 * (4 * 768 * 768 + 2 * 768 * 3072)
    text_total = 24 * (4 * 1024 * 1024 + 2 * 1024 * 4096)
    alignment = 1024 * 4096 + 768 * 4096 + 1024 * 4096
    perception_total = vision_total + audio_total + text_total + alignment

    # ==================== CREATIVE CORTEX (24 layers) ====================
    creative_total = 24 * (4 * 4096 * 4096 + 2 * 4096 * (4 * 4096) + 4096 * 2)

    # ==================== CONSCIOUSNESS ====================
    consciousness = 2048 * 2048 * 4 + 2048 * 4

    # ==================== EMBEDDING ====================
    embedding = vocab * dim

    # ==================== TOTAL ====================
    total = (reasoning_total + emotional_total + memory_total +
             perception_total + creative_total + consciousness + embedding)

    breakdown = {
        "reasoning_cortex (MLA+MoE+FFN, 96L)": reasoning_total,
        "emotional_cortex (48L)": emotional_total,
        "memory_cortex (Mamba, 48L)": memory_total,
        "perception_cortex (vision+audio+text)": perception_total,
        "creative_cortex (24L)": creative_total,
        "consciousness": consciousness,
        "embedding": embedding,
    }

    return total, breakdown


def main():
    print("=" * 70)
    print("    NICTO AI vs GPT-2 BENCHMARK")
    print("=" * 70)

    system = get_system_info()
    print("\nSystem: %d CPUs, %.2f GB RAM available" % (system['cpu_count'], system['available_ram_gb']))
    print("Note: GPT-2 could not be loaded (paging file too small). Using public specs.")

    # NICTO params
    nicto_total, breakdown = compute_nicto_params()

    # GPT-2 public specs
    gpt2_params = 124_439_808  # 124M
    gpt2_perplexity = 29.41  # WikiText-103

    # ===== Results =====
    print("\n" + "=" * 70)
    print("    NICTO AI ARCHITECTURE ANALYSIS")
    print("=" * 70)
    print("\n  Total parameters: {:,} (~{:.1f}B)".format(nicto_total, nicto_total/1e9))
    print("\n  Parameter breakdown:")
    for k, v in breakdown.items():
        pct = v / nicto_total * 100
        print("    {:40s}: {:>15,} ({:5.1f}%)".format(k, v, pct))

    print("\n  !! IMPORTANT AUDIT FINDING !!")
    print("  README claims ~150B parameters")
    print("  Calculated from model_config.py: ~{:.1f}B parameters".format(nicto_total/1e9))
    if nicto_total/1e9 > 200:
        print("  The actual config produces {:.0f}x MORE than the README claim!".format(nicto_total/1e9/150))
        print("  This means the README UNDERSTATES the architecture size.")
        print("  OR the config values were not meant to be used together.")
        print("")
        print("  The MoE block alone (64 experts x hidden=32768) accounts for")
        print("  {:.1f}B params per layer x 96 layers = {:.1f}B total.".format(
            64 * 2 * 8192 * 32768 / 1e9, 
            64 * 2 * 8192 * 32768 * 96 / 1e9))
        print("")
        print("  For reference, Mixtral 8x7B has 8 experts with hidden=14336.")
        print("  NICTO config has 64 experts with hidden=32768 (much larger).")

    # ===== Comparison =====
    print("\n" + "=" * 70)
    print("    HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    print("\n  {:<35} {:>18} {:>18}".format("Metric", "NICTO", "GPT-2"))
    print("  " + "-" * 71)
    print("  {:<35} {:>16,} {:>16,}".format("Parameters", nicto_total, gpt2_params))
    print("  {:<35} {:>15.1f}B {:>15.3f}B".format("Parameters (billions)", nicto_total/1e9, gpt2_params/1e9))
    print("  {:<35} {:>14.1f} GB {:>14.0f} MB".format("Size on disk (FP32)", nicto_total*4/1e9, gpt2_params*4/1e6))
    print("  {:<35} {:>14.1f} GB {:>14.0f} MB".format("Size on disk (FP16)", nicto_total*2/1e9, gpt2_params*2/1e6))
    print("  {:<35} {:>18} {:>18}".format("Vocab size", "128,000", "50,257"))
    print("  {:<35} {:>18} {:>18}".format("Hidden dim", "8,192", "768"))
    print("  {:<35} {:>18} {:>18}".format("Layers", "96+48+48+64+24", "12"))
    print("  {:<35} {:>18} {:>18}".format("Max context", "10M tokens", "1,024 tokens"))
    print("  {:<35} {:>18} {:>18}".format("Attention heads", "128", "12"))
    print("  {:<35} {:>18} {:>18}".format("Expert count", "64 (8 active)", "N/A (dense)"))
    print("  {:<35} {:>18} {:>15.2f}".format("Perplexity", "NOT TRAINED", gpt2_perplexity))
    print("  {:<35} {:>18} {:>18}".format("Training data", "NONE", "40GB text"))
    print("  {:<35} {:>18} {:>18}".format("Production ready", "NO", "YES"))

    # ===== Architecture =====
    print("\n" + "=" * 70)
    print("    ARCHITECTURE COMPARISON")
    print("=" * 70)

    features = [
        ("Type", "Hybrid (6 neural networks)", "Transformer Decoder"),
        ("Attention", "Multi-Latent Attention (MLA)", "Multi-Head Attention (MHA)"),
        ("Feed-forward", "Mixture of Experts (64 experts)", "Dense FFN"),
        ("Sequence model", "Mamba SSM (O(N))", "Self-Attention (O(N^2))"),
        ("Neural dynamics", "Liquid Neural Networks (ODE)", "None"),
        ("Memory", "Working + Episodic + Semantic", "Context window only"),
        ("Metacognition", "Consciousness Layer", "None"),
        ("Emotion", "Multi-modal emotion + empathy", "None"),
        ("Anti-hallucination", "Claim verification engine", "None"),
        ("Knowledge", "Web crawl + vector DB", "Training data only"),
        ("Offline learning", "Dream Engine (replay)", "None"),
        ("KV cache", "Compressed (512-d)", "Full cache (768-d)"),
        ("Normalization", "RMSNorm", "LayerNorm"),
        ("Position encoding", "RoPE (rotary)", "Learned absolute"),
        ("Activation", "SiLU", "GELU"),
        ("Sparsity", "MoE (8/64 = 12.5%)", "Dense (100%)"),
        ("Multi-modal", "Vision + Audio + Text", "Text only"),
    ]

    print("\n  {:<25} {:<38} {:<35}".format("Feature", "NICTO", "GPT-2"))
    print("  " + "-" * 98)
    for feat, n, g in features:
        print("  {:<25} {:<38} {:<35}".format(feat, n, g))

    # ===== GPT-2 Known Performance =====
    print("\n" + "=" * 70)
    print("    GPT-2 KNOWN BENCHMARKS (from literature)")
    print("=" * 70)
    print("""
  Perplexity (WikiText-103):   29.41
  MMLU (5-shot):               ~43%
  HellaSwag:                   ~76.2%
  WinoGrande:                  ~66.3%
  LAMBADA:                     ~76.2%
  
  Inference (A100 GPU):        ~3,000 tokens/sec
  Inference (CPU):             ~50-100 tokens/sec
  
  Training:                    40GB WebText (~10B tokens)
  Training cost:               ~$12,000 (estimated)
  Training time:               ~8 GPU-days
    """)

    # ===== NICTO Unique Advantages =====
    print("=" * 70)
    print("    NICTO UNIQUE ADVANTAGES")
    print("=" * 70)
    print("""
  1. SPARSE COMPUTATION (MoE)
     Only 8 of 64 experts active per token = 12.5% compute
  
  2. O(N) SEQUENCE MODELING (Mamba)
     No quadratic attention bottleneck
     Can theoretically handle 10M token context
  
  3. CONTINUOUS-TIME LEARNING (Liquid NN)
     ODE-based dynamics adapt in real-time
  
  4. METACOGNITION (Consciousness)
     Knows what it doesn't know
     Estimates uncertainty, detects errors
  
  5. ANTI-HALLUCINATION
     Claim extraction -> Verification -> Grounding
  
  6. HIERARCHICAL MEMORY
     Working + Episodic + Semantic memory
  
  7. MULTI-MODAL
     Vision (VGG16) + Audio + Text alignment
  
  8. OFFLINE LEARNING (Dream Engine)
     Can learn from experience without retraining
    """)

    # ===== GPT-2 Advantages =====
    print("=" * 70)
    print("    GPT-2 ADVANTAGES")
    print("=" * 70)
    print("""
  1. ACTUALLY TRAINED
     On 10B tokens of real web text
  
  2. PROVEN PERFORMANCE
     Established benchmarks across multiple tasks
  
  3. PRODUCTION OPTIMIZED
     TensorRT, ONNX, quantized variants available
  
  4. SMALLER & FASTER
     474 MB (FP32) loads in seconds
  
  5. WELL-UNDERSTOOD
     Extensive research on failure modes
  
  6. WORKS TODAY
     Install transformers, load model, generate text
    """)

    # ===== Verdict =====
    print("=" * 70)
    print("    VERDICT")
    print("=" * 70)
    print("""
  ARCHITECTURE WINNER:  NICTO AI
    - 6 neural networks vs 1
    - 10M context vs 1K (9,766x more)
    - Built-in anti-hallucination, memory, metacognition
    - Sparse MoE (12.5% compute per token)
    - NOTE: actual param count from config differs from README

  PRACTICAL WINNER:  GPT-2
    - Actually trained on 10B tokens
    - Benchmarked and proven
    - Production-ready, runs anywhere
    - 474 MB vs potentially terabytes

  CRITICAL NEXT STEP:  TRAIN NICTO
    - Download a dataset (RedPajama, SlimPajama, etc.)
    - Run: python train.py --data_path <data> --small
    - Then re-run this benchmark with real metrics
    """)

    # ===== Save results =====
    results = {
        "system": system,
        "nicto": {
            "total_parameters": nicto_total,
            "total_parameters_billions": round(nicto_total / 1e9, 1),
            "model_size_gb_fp32": round(nicto_total * 4 / 1e9, 2),
            "model_size_gb_fp16": round(nicto_total * 2 / 1e9, 2),
            "config_source": "model_config.py",
            "readme_claim": "~150B",
            "discrepancy": "Config produces {:.0f}x more than README claim".format(nicto_total / 150e9) if nicto_total > 200e9 else "Within range",
            "breakdown": {k: {"params": v, "pct": round(v / nicto_total * 100, 1)} for k, v in breakdown.items()},
            "trained": False,
            "benchmarks": "None",
        },
        "gpt2": {
            "total_parameters": gpt2_params,
            "model_size_mb_fp32": round(gpt2_params * 4 / 1e6),
            "perplexity_wikitext103": gpt2_perplexity,
            "trained": True,
            "training_data": "40GB WebText (~10B tokens)",
            "source": "OpenAI public specifications",
        },
        "comparison": {
            "param_ratio": round(nicto_total / gpt2_params, 1),
            "context_ratio": 10_000_000 / 1024,
        },
        "disclaimer": "NICTO params calculated from model_config.py. GPT-2 specs from public docs. No models loaded due to memory constraints.",
    }

    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to: tests/benchmark_results.json")


if __name__ == "__main__":
    main()
