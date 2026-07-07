# NICTO AI

## Overview

NICTO AI is an experimental architecture combining six neural subsystems — a
Mixture-of-Experts + Multi-Latent-Attention "reasoning" stack, a Liquid Neural
Network "emotional" stack, a Mamba state-space "memory" stack, a multimodal
perception block, a transformer "creative" block, and a meta-cognitive
"consciousness" layer — fused through a learned cross-attention "neural bus."

**Status: early-stage research/engineering project.** The full architecture
described below is implemented in PyTorch and runs correctly (forward pass,
loss computation, and autoregressive generation all verified), but it has
**not** been trained at the scale described in earlier drafts of this
document. What has actually been trained is the much smaller model in
`nicto_ai/training/model_train.py`.

### What's verified as of this update
- The 6-network architecture (`nicto_ai/core/model.py`) instantiates and runs
  a full forward + backward pass without errors, including generation.
- The training pipeline (`nicto_ai/training/train.py`) correctly loads real
  JSONL/text data and mixed-weighted datasets when given `--data-path` or
  `--data-mix` (previously these flags were parsed but silently ignored —
  training always fell back to synthetic random tokens regardless of what
  was passed; this is now fixed).
- Checkpoint resume (`--resume path/to/checkpoint.pt`) is now supported for
  warm-starting from a previous run instead of training from scratch.
- The voice pipeline (`nicto_ai/voice/`) correctly calls the Anthropic Claude
  API as its LLM backend; NICTO's own model is not yet wired in as a live
  chat backend (see `NICTOBackend` in `backend_interface.py`, which is an
  explicit stub pending a trained checkpoint).
- There is currently no automated test suite in this repository.

### What's aspirational / not yet true
- The "~150B parameters / ~20B active" scale in earlier documentation refers
  to the architecture's theoretical maximum configuration, not a trained
  model. No checkpoint at that scale exists.
- Benchmark numbers (SWE-bench, ARC-AGI-2, GPQA, MMLU) below are **targets**,
  not results. None have been run.
- Streaming audio, wake-word detection, and interrupt handling are not yet
  implemented in `nicto_ai/voice/`.


## Architecture: 6 Neural Networks

| Network | Parameters | Technology | Purpose |
|---------|------------|------------|---------|
| **Reasoning Cortex** | ~50B | DeepSeek MoE + MLA | Logic, math, code, reasoning |
| **Emotional Cortex** | ~30B | Liquid Neural Networks | Empathy, emotions, real-time adaptation |
| **Memory Cortex** | ~20B | Mamba SSM | 10M token context, hierarchical memory |
| **Perception Cortex** | ~25B | Multimodal Transformer | Vision, audio, text understanding |
| **Creative Cortex** | ~15B | Diffusion + Transformer | Generation, imagination |
| **Consciousness Layer** | ~10B | Meta-Cognitive | Self-awareness, intention |

**Total: ~150B parameters | Active: ~20B per token**

### Training Model (94M params)
For initial training on consumer GPUs, we use a scaled-down version:
- Dim: 1024, Heads: 8, MoE Experts: 4
- Fits on T4 16GB with gradient checkpointing
- Trains at ~6,000 tokens/sec

## Data Collection System

NICTO uses curated datasets from HuggingFace for training:

### Priority 1: Core Pretraining
| Dataset | Tokens | License | Use |
|---------|--------|---------|-----|
| FineWeb-Edu | 1.3T | ODC-By | Educational quality |
| SlimPajama | 627B | Apache 2.0 | General web |
| Wikipedia | 20B | CC-BY-SA-3.0 | Encyclopedia |

### Priority 2: Code & Math
| Dataset | Tokens | License | Use |
|---------|--------|---------|-----|
| The Stack (Python) | 50B | MIT | Code |
| OpenHermes 2.5 | 1B | OpenAI | Instructions |
| MATH | 100M | MIT | Math reasoning |

### Recommended Training Mix (10B tokens)
| Dataset | % | Why |
|---------|---|-----|
| FineWeb-Edu | 30% | Educational quality |
| SlimPajama | 20% | General web |
| Wikipedia | 10% | Factual knowledge |
| The Stack (Python) | 10% | Code ability |
| OpenHermes 2.5 | 10% | Instructions |
| MATH | 10% | Math reasoning |
| UltraChat | 10% | Conversations |

## Technologies Implemented

### 1. Multi-Latent Attention (MLA)
- From DeepSeek-V3
- 90%+ KV cache compression
- 10M+ token context window

### 2. Mixture of Experts (MoE)
- 64 experts, 8 activated per token
- Auxiliary-loss-free load balancing
- Massive scale with efficient inference

### 3. Mamba (State Space Models)
- O(N) linear complexity
- 7x faster than transformers at long sequences
- Selective state spaces

### 4. Liquid Neural Networks
- Real-time parameter adaptation
- Continuous-time dynamics (ODEs)
- Post-training learning capability

### 5. DeepSearch
- Iterative search-read-reason loops
- Multi-hop reasoning
- Knowledge-gap detection

### 6. Consciousness Simulation
- Self-model and intention formation
- Meta-cognition
- Identity and personality

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- PyTorch 2.4+

### Install Dependencies
```bash
pip install -e .
```

### For Development
```bash
pip install -e ".[dev]"
```

## Quick Start

### Download Training Data
```bash
# List available datasets
python -m nicto_ai.data.registry

# Download Priority 1 datasets
python -m nicto_ai.data.downloader --priority 1 --max-samples 100000

# Full pipeline (download + process)
python -m nicto_ai.data.collect --priority 1 --process
```

### Train NICTO
```bash
# Train with synthetic data (quick smoke test, no real dataset needed)
python -m nicto_ai.training.train --config colab

# Train with real data (a single file or directory of .txt/.jsonl)
python -m nicto_ai.training.train --config colab --data-path nicto_ai/data/processed/fineweb_edu.jsonl

# Train with the README's recommended weighted mix (requires collecting data first, see below)
python -m nicto_ai.training.train --config colab --use-real-data

# Resume/warm-start from an existing checkpoint
python -m nicto_ai.training.train --config colab --use-real-data --resume checkpoints/colab/final.pt
```

### Generate Text
```python
import torch
from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig

# Load model
config = NICTOTrainConfig()
model = NICTOTrainModel(config)
model.load_state_dict(torch.load("checkpoints/colab/final.pt")["model"])

# Generate
prompt = torch.randint(0, config.vocab_size, (1, 10))
output = model.generate(prompt, max_new_tokens=100)
```

## Project Structure
```
NICTO/
├── nicto_ai/
│   ├── core/
│   │   ├── mla.py            # Multi-Latent Attention
│   │   ├── moe.py            # Mixture of Experts
│   │   ├── mamba.py          # State Space Models
│   │   ├── liquid.py         # Liquid Neural Networks
│   │   ├── consciousness.py  # Consciousness Layer
│   │   ├── emotion.py        # Emotional Processing
│   │   └── memory.py         # Hierarchical Memory
│   ├── data/
│   │   ├── registry.py       # Dataset registry (14 datasets)
│   │   ├── downloader.py     # HuggingFace downloader
│   │   ├── processor.py      # Clean, filter, tokenize
│   │   ├── collect.py        # Main pipeline
│   │   └── mix_config.py     # Training mix configs
│   ├── gan/
│   │   ├── generator.py      # Style-based generator
│   │   ├── discriminator.py  # Spectral-norm discriminator
│   │   ├── loss.py           # R3GAN loss
│   │   └── trainer.py        # GAN training loop
│   ├── training/
│   │   ├── model_train.py    # Trainable model (94M params)
│   │   └── train.py          # Training script
│   └── voice/
│       ├── tts.py            # Text-to-Speech
│       ├── stt.py            # Speech-to-Text
│       └── agent_loop.py     # Voice interaction
├── k8s/
│   └── training.yaml         # Kubernetes config (8x A100)
└── README.md
```

## Training Status

### Current Progress
- **Architecture:** Fully implemented, verified to run (forward/backward/generate) at small scale.
- **Trained model:** Only the 94M-parameter `NICTOTrainModel` (`training/model_train.py`) has been trained, and only briefly, on synthetic data.
- **Data:** Real dataset download/processing pipeline exists (`nicto_ai/data/`) but has not yet been run end-to-end into a completed training job.
- **Hardware target:** Google Colab T4 (16GB) for the small config; `colab_large`/`k8s` configs exist but are untested at that scale.

### Next Steps
1. Run `python -m nicto_ai.data.collect --priority 1 --process` to fetch and process real Priority-1 datasets.
2. Train with `--use-real-data` (and `--resume` if warm-starting from an existing checkpoint) and confirm loss drops meaningfully on real data, not just synthetic tokens.
3. Add an automated test suite — none currently exists.
4. Only after real training results exist: re-run benchmarks (MMLU, ARC, HellaSwag, TruthfulQA) and report actual numbers, not targets.
5. Scale to larger configs only once the small-scale run is validated.

## Benchmark Targets (not yet measured)

These are aspirational targets. No benchmark runs have been completed against a trained NICTO checkpoint.

| Benchmark | Current SOTA | NICTO Target |
|-----------|--------------|--------------|
| SWE-bench | 80.8% | 90%+ |
| ARC-AGI-2 | 52.9% | 70%+ |
| GPQA | ~75% | 85%+ |
| MMLU | 90%+ | 80%+ |
| Context Window | 2M tokens | 10M tokens |

## Roadmap

- [x] Core architecture implementation (verified to run correctly at small scale)
- [x] MLA (Multi-Latent Attention)
- [x] MoE (Mixture of Experts)
- [x] Mamba (State Space Models)
- [x] Liquid Neural Networks
- [x] Consciousness Layer
- [x] Emotional Processing
- [x] Hierarchical Memory
- [x] Training pipeline, verified functional (94M param model, real-data wiring fixed)
- [x] Data collection system (14 datasets registered; not yet run end-to-end)
- [x] GAN (NICTO-GAN) — implemented, untested
- [x] Voice engine (TTS/STT/agent loop) — functional, backed by Claude API; NICTO's own model not yet wired as a live backend
- [ ] Automated test suite
- [ ] Real data training run with reported (not projected) loss curves
- [ ] Distributed training (k8s) — untested
- [ ] Evaluation benchmarks — not yet run
- [ ] Streaming audio / wake-word / interrupt handling in voice pipeline
- [ ] Open source release readiness review

## License

Apache-2.0

## Acknowledgments

- DeepSeek for MLA and MoE innovations
- MIT CSAIL for Liquid Neural Networks
- Albert Gu & Tri Dao for Mamba
- HuggingFace for dataset hosting
- The open-source AI community
