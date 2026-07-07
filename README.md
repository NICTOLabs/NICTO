# NICTO AI

Open-source neural architecture — reasoning, memory, emotion, and consciousness in one model.

**Status:** Undergoing training on Google Colab (T4 GPU). Current checkpoint is ~207M parameters. No benchmarks run yet.

## Install

```bash
git clone https://github.com/NICTOLabs/NICTO.git
cd NICTO
pip install -e .
```

## Commands

```bash
# Chat with NICTO
nicto chat

# Train
nicto train

# Show model info
nicto info
```

## How it's trained

NICTO is trained on Google Colab Free (T4 16GB) using PyTorch. The training pipeline supports synthetic data and real datasets from HuggingFace.

## Architecture

NICTO combines multiple neural approaches in a single model:

| Component | What it does |
|-----------|-------------|
| **Reasoning** | Attention + MoE blocks for logic, math, code |
| **Memory** | Bidirectional transformer for context retention |
| **Emotion** | Causal transformer for adaptive behavior |
| **Creativity** | Causal transformer for generation |
| **Consciousness** | Self-monitoring projection layer |
| **Fusion gate** | Learned weighting across all components |

## What's in the repo

```
nicto_ai/
├── core/           # Model components
├── training/       # Training model + loop
├── data/           # Dataset registry, downloader
├── voice/          # TTS, STT, agent loop
├── gan/            # GAN trainer
├── dream/          # Synthetic data generation
└── verification/   # Claim verification
```

## Requirements

- Python 3.10+
- PyTorch 2.0+

## License

Apache-2.0
