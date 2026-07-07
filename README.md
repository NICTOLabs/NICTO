# NICTO AI

## What is this?

NICTO is an experimental AI architecture built from scratch — code, training pipeline, model, everything. It is currently **undergoing training on Google Colab** (free tier, T4 GPU).

**Current state: early training.** NICTO has been trained to ~207M parameters on synthetic data. The architecture supports scaling to much larger configurations, but no large-scale checkpoint exists yet. No benchmarks have been run.

## Architecture

NICTO combines multiple neural network approaches in a single model:

- **Reasoning stack**: Attention + Mixture-of-Experts (MoE) blocks
- **Memory stack**: Bidirectional transformer layers
- **Emotional stack**: Causal transformer layers
- **Creative stack**: Causal transformer layers
- **Consciousness layer**: Self-monitoring projection
- **Fusion gate**: Learned weighting across all stacks

The trained checkpoint (207M params, dim=1024) is in this repo at `nicto_model_final.pt` via Git LFS.

## Training

Training runs on Google Colab Free (T4 16GB). The model trains on curated synthetic data and is being gradually exposed to real datasets (Wikipedia, code, reasoning).

```bash
# Train on Colab
python -m nicto_ai.training.train --config colab

# Train with real data mix
python -m nicto_ai.training.train --config colab --use-real-data

# Resume from checkpoint
python -m nicto_ai.training.train --config colab --resume checkpoints/colab/final.pt
```

## What works

- Full architecture implemented in PyTorch
- Forward pass, loss computation, autoregressive generation — all verified
- Training pipeline with real JSONL data support
- Checkpoint save/resume
- Data collection system (14 datasets from HuggingFace)
- GAN (NICTO-GAN) — implemented, untested
- Voice engine (TTS/STT/agent loop) — functional, uses local model

## What doesn't work yet

- Large-scale training (requires A100/H100, not available yet)
- Benchmarks (none run)
- Distributed training
- Streaming audio / wake-word detection
- Automated test suite

## Data collection

```bash
# List available datasets
python -m nicto_ai.data.registry

# Download and process
python -m nicto_ai.data.collect --priority 1 --process
```

## Project structure

```
nicto_ai/
├── core/           # Model components (attention, MoE, memory, emotion, consciousness)
├── training/       # Training model + training loop
├── data/           # Dataset registry, downloader, processor
├── voice/          # TTS, STT, agent loop, backend
├── gan/            # GAN generator, discriminator, trainer
├── dream/          # Dream engine, data generation
└── verification/   # Claim verification, grounding
```

## License

Apache-2.0
