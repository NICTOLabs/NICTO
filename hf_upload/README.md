---
language: en
license: other
license_name: All rights reserved (NICTOLabs)
tags:
  - nicto
  - transformer
  - mixture-of-experts
  - mla-attention
  - consciousness
  - text-generation
  - training
  - data-pipeline
pipeline_tag: text-generation
---

# NICTO AI Model

NICTO is a novel neural network architecture combining multiple advanced techniques:

- **Multi-Latent Attention (MLA)** - Efficient long-context processing based on DeepSeek-V3
- **Mixture of Experts (MoE)** - Sparse computation with top-2 routing
- **Consciousness Layer** - Metacognition and uncertainty estimation
- **Fusion Gate** - 4-way output fusion (Reasoning, Memory, Emotional, Creative)
- **Emotion System** - Emotional state tracking
- **Hierarchical Memory** - Working, episodic, and semantic memory

## Architecture

```
Input tokens
    |
    v
[Token Embedding + Position Embedding]
    |
    v
+---+---+---+---+
|   |   |   |   |
R   M   E   C   Co  (5 parallel streams)
|   |   |   |   |
+---+---+---+---+
    |
    v
[Fusion Gate] (weighted combination)
    |
    v
[Output Projection] -> logits
```

- **R (Reasoning)**: MLA attention + MoE FFN
- **M (Memory)**: Bidirectional self-attention
- **E (Emotional)**: Causal transformer
- **C (Creative)**: Causal transformer
- **Co (Consciousness)**: Self-monitoring projection

## Usage

```python
from model import NICTOTrainModel, NICTOTrainConfig

config = NICTOTrainConfig(
    vocab_size=32000,
    dim=1024,
    max_seq_len=1024,
    reasoning_layers=6,
    n_heads=8,
    n_kv_heads=2,
    moe_experts=4,
    moe_activated=2,
    memory_layers=4,
    emotional_layers=4,
    creative_layers=4,
)

model = NICTOTrainModel(config)
params = model.count_parameters()
print(f"Parameters: {params:,} ({params/1e6:.0f}M)")

# Generate text
import torch
ids = torch.randint(0, 32000, (1, 10))
output = model.generate(ids, max_new_tokens=50)
```

## Training

```python
# Quick training test
input_ids = torch.randint(0, 32000, (2, 128))
labels = torch.randint(0, 32000, (2, 128))
output = model(input_ids, labels=labels)
print(f"Loss: {output['loss'].item():.4f}")
```

## Data Collection System

NICTO includes a comprehensive data collection pipeline for training:

### Available Datasets

| Priority | Dataset | Tokens | License | Use |
|----------|---------|--------|---------|-----|
| 1 | FineWeb-Edu | 1.3T | ODC-By | Educational quality |
| 1 | SlimPajama | 627B | Apache 2.0 | General web |
| 1 | Wikipedia | 20B | CC-BY-SA-3.0 | Encyclopedia |
| 2 | The Stack (Python) | 50B | MIT | Code |
| 2 | OpenHermes 2.5 | 1B | OpenAI | Instructions |
| 2 | MATH | 100M | MIT | Math reasoning |
| 3 | UltraChat | 5B | MIT | Conversations |
| 3 | Dolly 15K | 10M | CC-BY-SA-3.0 | Instructions |

### Usage

```python
# List all datasets
python -m nicto_ai.data.registry

# Download Priority 1 datasets
python -m nicto_ai.data.downloader --priority 1 --max-samples 100000

# Full pipeline
python -m nicto_ai.data.collect --priority 1 --process
```

### Training Mix (10B tokens)

| Dataset | % | Why |
|---------|---|-----|
| FineWeb-Edu | 30% | Educational quality |
| SlimPajama | 20% | General web |
| Wikipedia | 10% | Factual knowledge |
| The Stack (Python) | 10% | Code ability |
| OpenHermes 2.5 | 10% | Instructions |
| MATH | 10% | Math reasoning |
| UltraChat | 10% | Conversations |

## Training Status

- **Model:** 94M parameters (trainable version)
- **Hardware:** Google Colab T4 (16GB)
- **Speed:** ~6,000 tokens/sec
- **Loss:** Dropping from 10.55 → 7.68 (step 300/2000)

## Configs

| Config | Params | Target |
|--------|--------|--------|
| Colab T4 | ~94M | T4 16GB |
| Colab Pro | ~300M | A100 40GB |
| Full Scale | ~1B | 8x A100 80GB |

## Citation

```bibtex
@software{nicto2026,
  title={NICTO AI: Neural Architecture with Consciousness and MoE},
  author={NICTO Labs},
  year={2026},
  url={https://github.com/NICTOLabs/-NICTO}
}
```
