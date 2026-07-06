# NICTO Data Collection System

## Overview

This module provides tools for collecting, processing, and preparing training data for NICTO AI. It supports downloading datasets from HuggingFace Hub, cleaning/filtering text, and creating mixed training datasets.

## Directory Structure

```
nicto_ai/data/
├── __init__.py          # Module exports
├── registry.py          # Dataset registry (all available datasets)
├── downloader.py        # Download from HuggingFace
├── processor.py         # Clean, filter, deduplicate, tokenize
├── collect.py           # Main collection pipeline
├── mix_config.py        # Data mixing configurations
├── raw/                 # Downloaded raw data (JSONL)
├── processed/           # Cleaned and filtered data
└── tokenized/           # Tokenized binary files
```

## Quick Start

### 1. List Available Datasets

```bash
python -m nicto_ai.data.registry
```

### 2. Download Data

```bash
# Download Priority 1 datasets (recommended starting point)
python -m nicto_ai.data.downloader --priority 1 --max-samples 100000

# Download specific dataset
python -m nicto_ai.data.downloader --dataset "FineWeb-Edu" --max-samples 50000

# Download all datasets
python -m nicto_ai.data.downloader --all
```

### 3. Process Data

```bash
# Process all raw files
python -m nicto_ai.data.processor --all

# Process specific file
python -m nicto_ai.data.processor --input raw/fineweb_edu.jsonl
```

### 4. Tokenize Data

```bash
# Tokenize all processed files
python -m nicto_ai.data.processor --all --tokenize

# Tokenize specific file
python -m nicto_ai.data.processor --input processed/fineweb_edu.jsonl --tokenize
```

### 5. Full Pipeline

```bash
# Download + process in one command
python -m nicto_ai.data.collect --priority 1 --max-samples 100000 --process
```

## Dataset Registry

### Priority 1: Core Pretraining

| Dataset | HuggingFace ID | Tokens | License | Use Case |
|---------|----------------|--------|---------|----------|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 1.3T | ODC-By | Educational quality web text |
| **SlimPajama** | `cerebras/SlimPajama-627B` | 627B | Apache 2.0 | Cleaned web text |
| **Wikipedia** | `wikimedia/wikipedia` | 20B | CC-BY-SA-3.0 | Encyclopedia knowledge |

### Priority 2: Code & Math

| Dataset | HuggingFace ID | Tokens | License | Use Case |
|---------|----------------|--------|---------|----------|
| **The Stack (Python)** | `bigcode/thestack` | 50B | MIT | Code ability |
| **OpenHermes 2.5** | `teknium/OpenHermes-2.5` | 1B | OpenAI | Instructions |
| **GSM8K** | `openai/gsm8k` | 10M | MIT | Grade school math |
| **MATH** | `hendrycks/competition_math` | 100M | MIT | Competition math |

### Priority 3: Knowledge & Reasoning

| Dataset | HuggingFace ID | Tokens | License | Use Case |
|---------|----------------|--------|---------|----------|
| **UltraChat** | `stingning/ultrachat` | 5B | MIT | Multi-turn conversations |
| **Dolly 15K** | `databricks/dolly-15k` | 10M | CC-BY-SA-3.0 | Instructions |
| **PubMed** | `pubmed_qa` | 200M | CC0 | Medical knowledge |

### Priority 4: Evaluation

| Dataset | HuggingFace ID | Tokens | License | Use Case |
|---------|----------------|--------|---------|----------|
| **MMLU** | `cais/mmlu` | 50M | MIT | Knowledge test |
| **ARC** | `allenai/arc` | 10M | CC-BY-4.0 | Reasoning |
| **HellaSwag** | `Rowan/hellaswag` | 10M | MIT | Commonsense |
| **TruthfulQA** | `truthfulqa/truthful_qa` | 5M | MIT | Truthfulness |

## Training Mix Recommendations

### For T4 (16GB VRAM)

| Config | Total Tokens | Training Time | Use Case |
|--------|--------------|---------------|----------|
| `t4_small` | 1B tokens | ~2 hours | Quick testing |
| `t4_medium` | 3B tokens | ~6 hours | Standard training |
| `t4_large` | 10B tokens | ~20 hours | Extended training |

### For k8s (8x A100)

| Config | Total Tokens | Training Time | Use Case |
|--------|--------------|---------------|----------|
| `k8s_full` | 100B tokens | ~1 week | Full training |

### Recommended Mix (10B tokens total)

| Dataset | Tokens | Percentage | Why |
|---------|--------|------------|-----|
| FineWeb-Edu | 3B | 30% | Educational quality |
| SlimPajama | 2B | 20% | General web |
| Wikipedia | 1B | 10% | Factual knowledge |
| The Stack (Python) | 1B | 10% | Code ability |
| OpenHermes 2.5 | 1B | 10% | Instructions |
| MATH | 1B | 10% | Math reasoning |
| UltraChat | 1B | 10% | Conversations |

## Using Real Data in Training

### Option 1: Single Dataset

```python
# In train.py config
train_config = {
    "data_path": "nicto_ai/data/processed/fineweb_edu.jsonl",
    # ... other config
}
```

### Option 2: Mixed Datasets

```python
# In train.py config
train_config = {
    "data_mix": [
        ("processed/fineweb_edu.jsonl", 0.30),
        ("processed/slimpajama.jsonl", 0.20),
        ("processed/wikipedia.jsonl", 0.10),
        ("processed/the_stack_python.jsonl", 0.10),
        ("processed/openhermes.jsonl", 0.10),
        ("processed/math.jsonl", 0.10),
        ("processed/ultrachat.jsonl", 0.10),
    ],
    # ... other config
}
```

### Option 3: Command Line

```bash
# Download and process data
python -m nicto_ai.data.collect --priority 1 --max-samples 100000 --process

# Train with real data
python -m nicto_ai.training.train --config colab --data-path nicto_ai/data/processed/fineweb_edu.jsonl
```

## Data Processing Pipeline

### 1. Download
- Streams from HuggingFace Hub
- Saves as JSONL format
- Supports max_samples limit

### 2. Clean
- Remove control characters
- Normalize whitespace
- Fix encoding issues

### 3. Filter
- Minimum/maximum length
- Quality checks (alphanumeric ratio, repetition, URLs)
- Field-based filtering (e.g., language for code)

### 4. Deduplicate
- Exact duplicate detection
- Near-duplicate detection (prefix-based)

### 5. Tokenize
- BPE tokenization (GPT-2 tokenizer)
- Saves as binary numpy arrays
- Supports chunked storage

## Colab Integration

For Google Colab training:

```python
# Download data in Colab
!python -m nicto_ai.data.collect --priority 1 --max-samples 50000 --process

# Train with real data
!python -m nicto_ai.training.train --config colab
```

## Troubleshooting

### Out of Memory
- Reduce `max_samples` in download
- Use smaller datasets first
- Enable gradient checkpointing

### Slow Downloads
- Use streaming mode (default)
- Limit `max_samples`
- Download one dataset at a time

### Poor Quality Data
- Increase `min_length` in processor
- Adjust quality filters
- Manual review of processed files

## Next Steps

1. **Download Priority 1 datasets** (FineWeb-Edu, SlimPajama, Wikipedia)
2. **Process and filter** for quality
3. **Create mixed training dataset** (10B tokens recommended)
4. **Train NICTO** with real data
5. **Evaluate** on MMLU, ARC, HellaSwag, TruthfulQA
6. **Iterate** based on results
