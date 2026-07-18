"""
NICTO 7B Scaling Guide
======================
How to take what we built on this CPU and scale to a competitive 7B model on cloud GPUs.

Architecture: NICTO v2 (decoder-only transformer, LLaMA-style)
Target: 7B parameters, competitive with LLaMA-2 7B
Hardware: 1-8x A100 80GB or 1-4x H100 80GB
"""

SCALING_GUIDE = """
# NICTO 7B — From CPU Validation to Production Training

## What We Built (CPU)

| Component | Status | Details |
|-----------|--------|---------|
| BPE Tokenizer | DONE | vocab=32000, trained on Wikipedia + GSM8K + SQuAD |
| Existing NICTO Model | DONE | 5.5M params, all 6 subsystems intact |
| Decoder-Only v2 | DONE | 128M params validated, LLaMA-style architecture |
| Data Pipeline | DONE | Memory-mapped .bin, mixed datasets |
| Training Loop | DONE | Grad accum, AMP, checkpointing, eval, generation |

## What You Need for 7B

### Hardware Requirements

| Setup | GPUs | VRAM | Cost/hr | Training Time (1T tokens) |
|-------|------|------|---------|---------------------------|
| 1x A100 80GB | 1 | 80GB | $1.10-1.64 | ~30 days |
| 4x A100 80GB | 4 | 320GB | $4.40-6.56 | ~8 days |
| 8x A100 80GB | 8 | 640GB | $8.80-13.12 | ~4 days |
| 1x H100 80GB | 1 | 80GB | $2.50-3.50 | ~12 days |
| 4x H100 80GB | 4 | 320GB | $10-14 | ~3 days |

**Recommended**: 4x A100 80GB on Lambda Labs ($4.40/hr) = ~8 days for 1T tokens.

### Cloud Providers

| Provider | GPU | Price/hr | Notes |
|----------|-----|----------|-------|
| Lambda Labs | A100 80GB | $1.10 | Best price/performance |
| Vast.ai | A100 80GB | $0.80-2.00 | Spot instances, variable |
| RunPod | A100 80GB | $1.64 | Good UX, community cloud |
| Azure ML | A100 80GB | $3.40 | Enterprise features |
| AWS p4d | A100 40GB | $3.27 | 8x A100 instances |

## Step-by-Step: Train NICTO 7B

### Step 1: Prepare Data (on your CPU)

```bash
# Download training data
python -m nicto_ai.data.downloader --dataset "Wikipedia" --max-samples 100000
python -m nicto_ai.data.downloader --dataset "FineWeb-Edu" --max-samples 100000

# Combine into single JSONL
python combine_data.py

# Retrain tokenizer on larger corpus
python -m nicto_ai.tokenizer.train --input combined_training_data.jsonl --vocab-size 32000

# Pre-tokenize to .bin (much faster training)
python -m nicto_ai.training.train_existing --pretokenize --data combined_training_data.jsonl
```

### Step 2: Set Up Cloud GPU

```bash
# On Lambda Labs (recommended)
# Instance: 4x A100 80GB, Ubuntu 22.04, CUDA 12.1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers datasets numpy tqdm

# Clone your repo
git clone https://github.com/NICTOLabs/NICTO.git
cd NICTO

# Copy tokenizer and data from your local machine
scp nicto_ai/tokenizer/artifacts/tokenizer.json <instance>:/NICTO/nicto_ai/tokenizer/artifacts/
scp combined_training_data.jsonl <instance>:/NICTO/
```

### Step 3: Train 7B Model

```bash
# Single GPU training
python -m nicto_ai.training.train_v2 \
    --model v2 \
    --size 7b \
    --config a100_80g \
    --data combined_training_data.jsonl

# Multi-GPU training (4x A100)
torchrun --nproc_per_node=4 \
    -m nicto_ai.training.train_v2 \
    --model v2 \
    --size 7b \
    --config a100_8gpus \
    --data combined_training_data.jsonl
```

### Step 4: Monitor Training

```bash
# W&B logging (optional)
python -m nicto_ai.training.train_v2 \
    --model v2 --size 7b --config a100_80g \
    --data combined_training_data.jsonl --wandb
```

## Architecture: NICTO v2 (7B Config)

```python
from nicto_ai.training.model_v2 import NICTOModel, config_7b

model = NICTOModel(config_7b())
# Parameters: 6,738,415,616 (6.74B)
# VRAM: ~28GB (fp16), ~14GB (int8)
```

### Config Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| dim | 4096 | Hidden dimension |
| n_heads | 32 | Query attention heads |
| n_kv_heads | 8 | KV heads (GQA, 4x compression) |
| n_layers | 32 | Transformer layers |
| ffn_dim | 11008 | SwiGLU FFN dimension |
| vocab_size | 32000 | BPE vocabulary |
| max_seq_len | 4096 | Maximum sequence length |

### VRAM Estimation

| Precision | Model | Optimizer | Total |
|-----------|-------|-----------|-------|
| fp32 | 27GB | 54GB | ~81GB |
| fp16/bf16 | 13.5GB | 27GB | ~41GB |
| int8 | 6.75GB | 13.5GB | ~21GB |

With gradient checkpointing: ~60% of above.

## MoE Variant (7B-MoE)

For higher capacity without proportionally more VRAM:

```python
from nicto_ai.training.model_v2 import NICTOModel, config_7b_moe

model = NICTOModel(config_7b_moe())
# Total params: ~12B, Active per token: ~3B
# VRAM: ~41GB (fp16)
```

## Looped Variant (7B-Looped)

Ouro-style reasoning with 4 recurrent steps:

```python
from nicto_ai.training.model_v2 import NICTOModel, config_7b_looped

model = NICTOModel(config_7b_looped())
# Parameters: ~7B, Equivalent computation: ~28B
# VRAM: ~28GB (fp16)
```

## Data Requirements

| Tokens | Quality | Notes |
|--------|---------|-------|
| 100B | Minimum | Coherent text, basic reasoning |
| 300B | Good | Competitive with GPT-2/3 |
| 1T | Great | Competitive with LLaMA-2 7B |
| 2T+ | Best | State-of-the-art |

### Recommended Data Mix

| Dataset | Weight | Tokens (1T total) | Purpose |
|---------|--------|-------------------|---------|
| FineWeb-Edu | 30% | 300B | Educational web text |
| SlimPajama | 20% | 200B | General web text |
| Wikipedia | 10% | 100B | Encyclopedia knowledge |
| The Stack (Python) | 10% | 100B | Code understanding |
| OpenHermes 2.5 | 10% | 100B | Instruction following |
| MATH | 10% | 100B | Mathematical reasoning |
| UltraChat | 10% | 100B | Multi-turn conversations |

### Download Full Data

```bash
# Download all Priority 1+2 datasets
python -m nicto_ai.data.collect --priority 1 --max-samples 1000000
python -m nicto_ai.data.collect --priority 2 --max-samples 1000000

# Process and combine
python combine_data.py --input nicto_ai/data/raw/ --output full_training_data.jsonl
```

## Training Hyperparameters (7B)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-4 | Peak LR |
| Min LR | 1e-5 | End of cosine schedule |
| Warmup steps | 500 | Linear warmup |
| Total steps | 100,000 | ~1T tokens |
| Batch size | 4 | Per GPU |
| Grad accum | 4 | Effective batch = 16 |
| Gradient clip | 1.0 | Max norm |
| Weight decay | 0.1 | AdamW |
| Optimizer | AdamW | betas=(0.9, 0.95) |
| Mixed precision | bf16 | On A100/H100 |

## Timeline

| Phase | Duration | Details |
|-------|----------|---------|
| Data prep | 1-2 days | Download, process, tokenize |
| Setup | 0.5 days | Cloud GPU, install deps |
| Training | 3-30 days | Depending on GPU count |
| Eval | 1 day | Benchmarks, generation tests |
| **Total** | **5-33 days** | |

## Quick Start (Copy-Paste)

```bash
# On your CPU (data prep)
python -m nicto_ai.tokenizer.train --input combined_training_data.jsonl
python combine_data.py

# On cloud GPU (4x A100)
pip install torch tokenizers datasets
git clone https://github.com/NICTOLabs/NICTO.git && cd NICTO
# Copy tokenizer.json and combined_training_data.jsonl

python -m nicto_ai.training.train_v2 \
    --model v2 --size 7b --config a100_8gpus \
    --data combined_training_data.jsonl --wandb
```
"""

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(SCALING_GUIDE)
