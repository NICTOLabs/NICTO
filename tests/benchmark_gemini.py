"""
NICTO AI vs Gemini Benchmark Comparison
Compares architecture, specs, and capabilities
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
    """Compute NICTO parameters from model_config.py"""
    dim = 8192
    vocab = 128_000
    head_dim = dim // 128

    # Reasoning Cortex (96 layers)
    mla_per_layer = (dim * 1536 + 1536 * (128 * head_dim) +
                     dim * (512 + 16 * head_dim) + 512 * (16 * head_dim) +
                     (128 * head_dim) * dim + dim + head_dim * 2)
    moe_per_layer = 64 * (2 * dim * 32768) + dim * 64
    reasoning_per_layer = mla_per_layer + moe_per_layer + dim
    reasoning_total = 96 * reasoning_per_layer

    # Emotional Cortex (48 layers)
    emotional_total = 48 * (4 * dim * dim + 2 * dim * (4 * dim) + dim * 2)

    # Memory Cortex (48 layers, Mamba)
    mamba_d_model = 4096
    mamba_d_state = 256
    mamba_inner = mamba_d_model * 2
    mamba_block = (mamba_d_model * mamba_inner * 2 + mamba_inner * 4 +
                   mamba_inner * mamba_d_state + mamba_inner +
                   mamba_inner * mamba_d_model)
    memory_total = 48 * (mamba_block + mamba_d_model * 2)

    # Perception Cortex
    vision_total = 24 * (4 * 1024 * 1024 + 2 * 1024 * 4096)
    audio_total = 16 * (4 * 768 * 768 + 2 * 768 * 3072)
    text_total = 24 * (4 * 1024 * 1024 + 2 * 1024 * 4096)
    alignment = 1024 * 4096 + 768 * 4096 + 1024 * 4096
    perception_total = vision_total + audio_total + text_total + alignment

    # Creative Cortex (24 layers)
    creative_total = 24 * (4 * 4096 * 4096 + 2 * 4096 * (4 * 4096) + 4096 * 2)

    # Consciousness
    consciousness = 2048 * 2048 * 4 + 2048 * 4

    # Embedding
    embedding = vocab * dim

    total = (reasoning_total + emotional_total + memory_total +
             perception_total + creative_total + consciousness + embedding)

    active_per_token = int(reasoning_total * 8 / 64)  # 8/64 experts active

    breakdown = {
        "reasoning_cortex": {"params": reasoning_total, "pct": round(reasoning_total / total * 100, 1)},
        "emotional_cortex": {"params": emotional_total, "pct": round(emotional_total / total * 100, 1)},
        "memory_cortex": {"params": memory_total, "pct": round(memory_total / total * 100, 1)},
        "perception_cortex": {"params": perception_total, "pct": round(perception_total / total * 100, 1)},
        "creative_cortex": {"params": creative_total, "pct": round(creative_total / total * 100, 1)},
        "consciousness": {"params": consciousness, "pct": round(consciousness / total * 100, 1)},
        "embedding": {"params": embedding, "pct": round(embedding / total * 100, 1)},
    }

    return total, active_per_token, breakdown


def main():
    print("=" * 72)
    print("       NICTO AI vs GEMINI BENCHMARK")
    print("=" * 72)

    system = get_system_info()
    print("\nSystem: %d CPUs, %.2f GB RAM" % (system['cpu_count'], system['available_ram_gb']))

    nicto_total, nicto_active, nicto_breakdown = compute_nicto_params()

    # Gemini specs (from official sources and industry analysis)
    # Google does NOT disclose exact parameter counts
    gemini = {
        "name": "Gemini 2.5 Pro",
        "release": "March 2025 (GA June 2025)",
        "architecture": "Sparse MoE Transformer (decoder-only)",
        "total_params_est": "1.5-1.8T (estimated, not disclosed)",
        "active_params_est": "~50-100B per token (estimated)",
        "context_window": 1_048_576,
        "max_output": 65_536,
        "vocab_size": "Not disclosed",
        "layers": "Not disclosed",
        "hidden_size": "Not disclosed",
        "attention_heads": "Not disclosed",
        "experts": "Not disclosed (MoE confirmed)",
        "modalities": ["Text", "Image", "Audio", "Video", "PDF"],
        "training_data": "Not disclosed (web, code, images, audio, video)",
        "thinking": True,
        "native_multimodal": True,
        "open_weights": False,
        "pricing_input": "$1.25 / 1M tokens",
        "pricing_output": "$10.00 / 1M tokens",
        "benchmarks": {
            "swe_bench_verified": "59.6%",
            "gpqa_diamond": "86.4%",
            "mmmu": "68%",
        },
        "source": "Google DeepMind technical report + model card",
    }

    # ===== NICTO Specs =====
    print("\n" + "=" * 72)
    print("       NICTO AI ARCHITECTURE")
    print("=" * 72)
    print("\n  Total parameters:     {:,} (~{:.1f}B)".format(nicto_total, nicto_total / 1e9))
    print("  Active per token:     {:,} (~{:.1f}B)".format(nicto_active, nicto_active / 1e9))
    print("  Size (FP16):          {:.1f} GB".format(nicto_total * 2 / 1e9))
    print("\n  Parameter breakdown:")
    for k, v in nicto_breakdown.items():
        print("    {:30s}: {:>15,} ({:5.1f}%)".format(k, v["params"], v["pct"]))

    # ===== Gemini Specs =====
    print("\n" + "=" * 72)
    print("       GEMINI 2.5 PRO ARCHITECTURE")
    print("=" * 72)
    print("\n  Total parameters:     {} (Google does NOT disclose)".format(gemini["total_params_est"]))
    print("  Active per token:     {} (estimated)".format(gemini["active_params_est"]))
    print("  Context window:       {:,} tokens".format(gemini["context_window"]))
    print("  Max output:           {:,} tokens".format(gemini["max_output"]))
    print("  Architecture:         {}".format(gemini["architecture"]))
    print("  Modalities:           {}".format(", ".join(gemini["modalities"])))
    print("  Thinking (CoT):       Built-in, configurable budget")
    print("  Native multimodal:    Yes (single Transformer stack)")
    print("  Open weights:         No (API only)")

    # ===== Head-to-Head =====
    print("\n" + "=" * 72)
    print("       HEAD-TO-HEAD COMPARISON")
    print("=" * 72)

    print("\n  {:<35} {:>18} {:>18}".format("Metric", "NICTO", "Gemini 2.5 Pro"))
    print("  " + "-" * 71)
    print("  {:<35} {:>16,} {:>18}".format("Parameters (total)", nicto_total, "~1,500,000,000,000"))
    print("  {:<35} {:>15.1f}B {:>18}".format("Parameters (total B)", nicto_total/1e9, "~1,500B (est.)"))
    print("  {:<35} {:>15.1f}B {:>18}".format("Active params/token", nicto_active/1e9, "~50-100B (est.)"))
    print("  {:<35} {:>14.1f} GB {:>18}".format("Model size (FP16)", nicto_total*2/1e9, "~3,000 GB (est.)"))
    print("  {:<35} {:>18} {:>18}".format("Architecture", "Hybrid MoE+Mamba+Liquid", "Sparse MoE Transformer"))
    print("  {:<35} {:>18} {:>18}".format("Hidden dim", "8,192", "Not disclosed"))
    print("  {:<35} {:>18} {:>18}".format("Layers", "96+48+48+64+24", "Not disclosed"))
    print("  {:<35} {:>18} {:>16,}".format("Max context", "10M tokens", gemini["context_window"]))
    print("  {:<35} {:>18} {:>16,}".format("Max output", "N/A", gemini["max_output"]))
    print("  {:<35} {:>18} {:>18}".format("Attention", "MLA (compressed KV)", "Standard MHA (MoE)"))
    print("  {:<35} {:>18} {:>18}".format("Sequence modeling", "Mamba SSM (O(N))", "Self-Attention (O(N^2))"))
    print("  {:<35} {:>18} {:>18}".format("Sparsity", "MoE 8/64 (12.5%)", "MoE (ratio unknown)"))
    print("  {:<35} {:>18} {:>18}".format("Multi-modal", "Vision+Audio+Text", "Vision+Audio+Video+Text+PDF"))
    print("  {:<35} {:>18} {:>18}".format("Thinking/CoT", "DeepSearch module", "Built-in thinking"))
    print("  {:<35} {:>18} {:>18}".format("Metacognition", "Consciousness layer", "None (public)"))
    print("  {:<35} {:>18} {:>18}".format("Memory system", "Working+Episodic+Semantic", "Context window only"))
    print("  {:<35} {:>18} {:>18}".format("Anti-hallucination", "Verification engine", "None (public)"))
    print("  {:<35} {:>18} {:>18}".format("Offline learning", "Dream Engine", "None"))
    print("  {:<35} {:>18} {:>18}".format("Normalization", "RMSNorm", "Not disclosed"))
    print("  {:<35} {:>18} {:>18}".format("Position encoding", "RoPE", "Not disclosed"))
    print("  {:<35} {:>18} {:>18}".format("KV cache", "Compressed (512-d)", "Standard"))
    print("  {:<35} {:>18} {:>18}".format("Training data", "NONE", "Massive multimodal corpus"))
    print("  {:<35} {:>18} {:>18}".format("Trained", "NO", "YES"))
    print("  {:<35} {:>18} {:>18}".format("Production ready", "NO", "YES (API)"))
    print("  {:<35} {:>18} {:>18}".format("Open weights", "Yes (if trained)", "No"))

    # ===== Gemini Benchmarks =====
    print("\n" + "=" * 72)
    print("       GEMINI 2.5 PRO BENCHMARKS (from Google)")
    print("=" * 72)
    print("""
  SWE-Bench Verified:    59.6%  (real-world coding)
  GPQA Diamond:          86.4%  (graduate-level science)
  MMMU:                  68.0%  (multimodal understanding)
  
  LMArena:               #1     (human preference ranking)
  
  Context utilization:   Effective up to ~300K-500K tokens
                         (1M theoretical, some degradation)
  
  Pricing:
    Input:   $1.25 / 1M tokens
    Output:  $10.00 / 1M tokens
    """)

    # ===== NICTO Advantages =====
    print("=" * 72)
    print("       NICTO ADVANTAGES OVER GEMINI")
    print("=" * 72)
    print("""
  1. OPEN WEIGHTS
     NICTO can be self-hosted, fine-tuned, modified
     Gemini is API-only, no weight access

  2. UNIQUE ARCHITECTURE (6 neural networks)
     - Mamba SSM: O(N) vs Gemini's O(N^2) attention
     - Liquid Neural Networks: ODE-based continuous dynamics
     - Consciousness Layer: metacognition, uncertainty
     - Hierarchical Memory: working + episodic + semantic
     - Anti-Hallucination: claim verification engine
     - Dream Engine: offline learning from experience

  3. LONGER THEORETICAL CONTEXT
     NICTO: 10M tokens (Mamba enables this)
     Gemini: 1M tokens (attention-based limit)

  4. SPARSER COMPUTATION
     NICTO: 12.5% of params active (8/64 experts)
     Gemini: MoE but ratio not disclosed

  5. TRANSPARENCY
     NICTO: Full architecture documented in code
     Gemini: Most specs hidden (params, layers, etc.)

  6. NO API COSTS
     NICTO: Self-hosted, no per-token charges
     Gemini: $1.25/$10.00 per 1M tokens
    """)

    # ===== Gemini Advantages =====
    print("=" * 72)
    print("       GEMINI ADVANTAGES OVER NICTO")
    print("=" * 72)
    print("""
  1. ACTUALLY TRAINED
     On massive multimodal corpus (web, code, video, audio)
     NICTO has zero training data

  2. PROVEN BENCHMARKS
     SWE-Bench 59.6%, GPQA 86.4%, MMMU 68%
     NICTO has no benchmark results

  3. NATIVE MULTIMODAL (5 modalities)
     Text + Image + Audio + Video + PDF
     NICTO: Text + Vision + Audio (3)

  4. PRODUCTION-READY API
     99.9% uptime, global infrastructure
     NICTO: Cannot even be instantiated (OOM)

  5. THINKING/REASONING BUILT-IN
     Configurable chain-of-thought budget
     NICTO: DeepSearch module (untested)

  6. MASSIVE SCALE
     Google's TPU infrastructure
     Trained on orders of magnitude more data

  7. CONTINUOUS UPDATES
     Regular model improvements
     NICTO: Static codebase
    """)

    # ===== Verdict =====
    print("=" * 72)
    print("       VERDICT")
    print("=" * 72)
    print("""
  ARCHITECTURE WINNER:  NICTO AI (with caveats)
    - More diverse neural network types (6 vs 1 Transformer)
    - O(N) sequence modeling via Mamba
    - Built-in metacognition, memory, anti-hallucination
    - Open weights, self-hostable
    - BUT: config produces ~3.3T params (README says 150B)
    - AND: nothing has been trained or tested

  PRACTICAL WINNER:  Gemini 2.5 Pro (by far)
    - Actually trained and deployed
    - Proven benchmarks across multiple domains
    - 5-modality native multimodal
    - Production API with 1M context
    - Google-scale infrastructure

  HONEST ASSESSMENT:
    NICTO is an UNTRAINED architecture experiment.
    Gemini is a DEPLOYED, PROVEN, PRODUCTION system.
    The gap is not architectural - it's empirical.
    NICTO needs training data + GPU time to be comparable.

  CRITICAL NEXT STEPS FOR NICTO:
    1. Fix config discrepancy (3.3T vs claimed 150B)
    2. Download training data (RedPajama, Dolma, etc.)
    3. Train on available GPU(s)
    4. Run benchmarks against Gemini's published scores
    5. Deploy and measure real-world performance
    """)

    # ===== Save results =====
    results = {
        "system": system,
        "nicto": {
            "total_parameters": nicto_total,
            "total_parameters_billions": round(nicto_total / 1e9, 1),
            "active_parameters_per_token": nicto_active,
            "active_parameters_billions": round(nicto_active / 1e9, 1),
            "model_size_gb_fp16": round(nicto_total * 2 / 1e9, 1),
            "max_context": "10M tokens (Mamba)",
            "architecture": "Hybrid (MLA + MoE + Mamba + Liquid + Consciousness)",
            "modalities": ["Text", "Vision (VGG16)", "Audio"],
            "trained": False,
            "config_source": "model_config.py",
            "readme_claim": "~150B",
            "actual_calculated": "~{:.1f}B".format(nicto_total / 1e9),
            "breakdown": nicto_breakdown,
        },
        "gemini": {
            "name": "Gemini 2.5 Pro",
            "release": "March 2025 (GA June 2025)",
            "total_parameters": "Not disclosed (estimated 1.5-1.8T)",
            "active_parameters": "Not disclosed (estimated 50-100B)",
            "context_window": 1_048_576,
            "max_output": 65_536,
            "architecture": "Sparse MoE Transformer",
            "modalities": ["Text", "Image", "Audio", "Video", "PDF"],
            "thinking": True,
            "native_multimodal": True,
            "open_weights": False,
            "trained": True,
            "benchmarks": {
                "swe_bench_verified": "59.6%",
                "gpqa_diamond": "86.4%",
                "mmmu": "68%",
            },
            "pricing": {
                "input": "$1.25 / 1M tokens",
                "output": "$10.00 / 1M tokens",
            },
        },
        "comparison": {
            "nicto_advantages": [
                "Open weights (self-hostable, fine-tunable)",
                "6 diverse neural networks (not just Transformer)",
                "O(N) sequence modeling via Mamba SSM",
                "Built-in metacognition (Consciousness layer)",
                "Hierarchical memory (working + episodic + semantic)",
                "Anti-hallucination verification engine",
                "Offline learning via Dream Engine",
                "10M theoretical context (vs 1M)",
                "12.5% sparse computation (MoE)",
            ],
            "gemini_advantages": [
                "Actually trained on massive multimodal corpus",
                "Proven benchmarks (SWE-Bench 59.6%, GPQA 86.4%)",
                "5-modality native multimodal",
                "Production-ready API with 99.9% uptime",
                "Built-in thinking/reasoning with budget control",
                "Google-scale training infrastructure",
                "Continuous updates and improvements",
                "Working today, no setup required",
            ],
        },
        "disclaimer": "Gemini parameter counts are NOT disclosed by Google. Estimates from industry analysis. NICTO params calculated from model_config.py.",
    }

    output_file = Path(__file__).parent / "benchmark_gemini_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to: tests/benchmark_gemini_results.json")


if __name__ == "__main__":
    main()
