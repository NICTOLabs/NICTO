"""
NICTO AI Training Configuration
Creates a ~1B param model that fits on a T4 GPU (16GB VRAM)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# TRAINING CONFIGS
# ============================================================

# Config for Google Colab Free (T4 16GB)
COLAB_T4_CONFIG = {
    "name": "nicto-1b-colab",
    "description": "~1B params, fits on T4 with gradient checkpointing",
    "model": {
        "vocab_size": 32000,
        "dim": 1024,
        "max_seq_len": 2048,
        "n_heads": 8,
        "n_kv_heads": 2,
        "moe_experts": 4,
        "moe_activated": 2,
        "moe_hidden": 2048,
        "mamba_d_state": 64,
        "mamba_layers": 6,
        "reasoning_layers": 6,
        "emotional_layers": 4,
        "memory_layers": 4,
        "creative_layers": 4,
    },
    "training": {
        "batch_size": 4,
        "gradient_accumulation": 8,  # effective batch = 32
        "learning_rate": 3e-4,
        "min_lr": 1e-5,
        "warmup_steps": 100,
        "total_steps": 5000,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "mixed_precision": "bf16",  # T4 supports bf16
        "gradient_checkpointing": True,
        "save_every": 500,
        "eval_every": 250,
    },
    "data": {
        "source": "openwebtext",
        "seq_len": 2048,
        "num_workers": 2,
    },
    "estimated_params": "~1B",
    "estimated_vram": "~12GB (with optimization)",
    "estimated_time": "~8-12 hours for 5000 steps",
}

# Config for Kubernetes (multi-GPU)
K8S_CONFIG = {
    "name": "nicto-3b-k8s",
    "description": "~3B params, distributed training on k8s",
    "model": {
        "vocab_size": 32000,
        "dim": 2048,
        "max_seq_len": 4096,
        "n_heads": 16,
        "n_kv_heads": 4,
        "moe_experts": 8,
        "moe_activated": 2,
        "moe_hidden": 4096,
        "mamba_d_state": 128,
        "mamba_layers": 12,
        "reasoning_layers": 12,
        "emotional_layers": 8,
        "memory_layers": 8,
        "creative_layers": 8,
    },
    "training": {
        "batch_size": 8,
        "gradient_accumulation": 4,
        "learning_rate": 3e-4,
        "min_lr": 1e-5,
        "warmup_steps": 200,
        "total_steps": 50000,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "save_every": 2000,
        "eval_every": 500,
        "distributed": True,
        "world_size": 4,  # 4 GPUs
    },
    "data": {
        "source": "redpajama-1t",
        "seq_len": 4096,
        "num_workers": 8,
    },
    "estimated_params": "~3B",
    "estimated_vram": "~40GB per GPU",
    "estimated_time": "~3-5 days for 50000 steps",
}


def create_model_from_config(config):
    """Create NICTO model from config dict"""
    from nicto_ai.core.model import NICTOModel
    from nicto_ai.core.mla import MultiLatentAttention
    from nicto_ai.core.moe import MixtureOfExperts
    from nicto_ai.core.mamba import MambaStack
    from nicto_ai.core.liquid import LiquidStack
    from nicto_ai.core.memory import HierarchicalMemory
    from nicto_ai.core.consciousness import RealConsciousnessLayer
    import torch.nn as nn

    mc = config["model"]
    dim = mc["dim"]

    model = NICTOModel(
        vocab_size=mc["vocab_size"],
        dim=dim,
        max_seq_len=mc["max_seq_len"],
    )

    # Override with training-friendly config
    model.reasoning_attention = MultiLatentAttention(
        dim=dim,
        n_heads=mc["n_heads"],
        n_kv_heads=mc["n_kv_heads"],
        kv_lora_rank=128,
        q_lora_rank=256,
        max_seq_len=mc["max_seq_len"],
    )
    model.reasoning_moe = MixtureOfExperts(
        dim=dim,
        n_experts=mc["moe_experts"],
        n_activated=mc["moe_activated"],
        hidden_dim=mc["moe_hidden"],
    )
    model.reasoning_ffn = nn.Sequential(
        nn.Linear(dim, mc["moe_hidden"]),
        nn.SiLU(),
        nn.Linear(mc["moe_hidden"], dim),
    )
    model.memory_mamba = MambaStack(
        n_layers=mc["mamba_layers"],
        d_model=dim,
        d_state=mc["mamba_d_state"],
    )

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("NICTO AI Training Configurations")
    print("=" * 60)

    for name, config in [("Colab T4", COLAB_T4_CONFIG), ("K8s Multi-GPU", K8S_CONFIG)]:
        print(f"\n--- {name} ---")
        print(f"  Model: {config['estimated_params']} params")
        print(f"  VRAM: {config['estimated_vram']}")
        print(f"  Time: {config['estimated_time']}")
        print(f"  Config: {config['name']}")

        model = create_model_from_config(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Actual params: {params:,} ({params/1e9:.2f}B)")
