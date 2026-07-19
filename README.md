# NICTO

**Neural Integrated Cognitive Transformer Architecture** — a cognitive language model with parallel reasoning paths, memory, emotion, creativity, and self-monitoring, fused into one unified system with a reward-guided cognitive executive.

## Architecture

NICTO is not just a transformer. Every forward pass runs **12 integrated components** in parallel, fused by learned gating:

```
Input (text + optional images + optional audio)
  → Multimodal Encoders
  → NOVA Core × N layers:
      [SSM + SparseAttn + MoE + MoD + PRS] fused by learned gate
  → Looped Reasoning (recurrent with exit gates)
  → Cognitive Subsystems (parallel):
      Memory · Emotional · Creative · Consciousness · HierarchicalMemory
  → NeuralBus (cross-network priority attention)
  → DeepSearch (beam-search for hard tokens)
  → Meta-Fusion Gate (learned weights over ALL outputs)
  → Reward System (multi-dimensional scoring of subsystem proposals)
  → Cognitive Executive / Top Model (reward-weighted cross-attention reasoning)
  → Next token
```

### Two-Tier Architecture

```
┌─────────────────────────────────────────────────────┐
│  TOP MODEL: Cognitive Executive                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Per-subsystem Proposal Heads                     │ │
│  │ Uncertainty-Aware Gating                         │ │
│  │ Self-Attention Reasoning Layers                  │ │
│  │ Residual Connection                              │ │
│  └─────────────────────────────────────────────────┘ │
│         ↑ Reward weights ↑ Uncertainty scores         │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Reward System                                    │ │
│  │ Coherence · Relevance · Creativity · Safety      │ │
│  └─────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│  BOTTOM STACK: NOVA + Subsystems                     │
│  NOVA Core → Looped → 5 Subsystems → NeuralBus →    │
│  DeepSearch → Meta-Fusion Gate                       │
└─────────────────────────────────────────────────────┘
```

### Components

| Component | What it does |
|-----------|-------------|
| **NOVA Core** | 5 parallel paths per block: SSM (Mamba-3), Sparse Attention, MoE, Mixture-of-Depth, Persistent Recurrent State |
| **Looped Reasoning** | Ouro-style recurrent block up to 32 steps with learned exit gates |
| **MoD** | Per-token router skips easy tokens — saves FLOPs dynamically |
| **NeuralBus** | Cross-network attention where each subsystem attends to the others |
| **DeepSearch** | Multi-step beam search over future tokens for uncertain positions |
| **Meta-Fusion Gate** | `softmax(W · concat[core, mem, emo, cre, con])` — learned per-token subsystem weighting |
| **Reward System** | Scores each subsystem on coherence, relevance, creativity, safety; produces softmax weights |
| **Cognitive Executive** | Top model: receives reward-weighted proposals from all subsystems, applies uncertainty gating, runs self-attention reasoning, outputs final logits |

### Config Scales

| Config | Params | dim | layers | MoE experts | looped steps | top model layers |
|--------|--------|-----|--------|-------------|-------------|-----------------|
| Tiny | 8.3M | 128 | 2 | 2 | 4 | 1 |
| 100M | 632M | 768 | 12 | 4 | 8 | 2 |
| 1B | ~1B | 2048 | 24 | 8 | 12 | 2 |
| 7B | ~7B | 4096 | 32 | 8 | 16 | 3 |
| 5T | ~5T | 8192 | 96 | **512** | **32** | 4 |

## Quick Start

```bash
git clone https://github.com/NICTOLabs/NICTO.git
cd NICTO

# Install dependencies
pip install -e .

# Train the tiny model (runs on CPU)
python train_master.py --config tiny --steps 1000

# Download more training data
python prepare_data.py --all

# Validate the architecture
python -m nicto_ai.training.model_master
```

## Data Pipeline

Pre-tokenized memory-mapped `.bin` files for O(1) random access during training:

```bash
# Download and pre-tokenize datasets
python prepare_data.py --datasets oasst1,dolly,c4,alpaca

# Train with auto-discovered data
python train_master.py --config 100m --steps 5000 --batch-size 2
```

Supported datasets: OASST1, Dolly 15K, C4, Alpaca, OpenOrca, FineWeb, Wikipedia, Cosmopedia.

## Project Structure

```
nicto_ai/
├── training/          # Models + training pipeline
│   ├── model_master.py       # Ultimate integration (all 10 components)
│   ├── model_multimodal.py   # Vision + Audio encoders
│   ├── model_nova.py         # NOVA core blocks
│   ├── model_unified.py      # NOVA + subsystems + fusion
│   ├── model_v2.py           # Decoder backbone
│   ├── data_pipeline_v2.py   # Memory-mapped .bin data pipeline
│   ├── trainer.py            # Training utilities
│   └── pretrain.py           # Pre-training entry point
├── core/              # Foundational components
│   ├── memory.py             # HierarchicalMemory (4-level)
│   ├── consciousness.py      # RealConsciousnessLayer
│   ├── deepsearch.py         # DeepSearchModule
│   ├── moe.py                # MoE with load balancing
│   ├── mamba.py              # Selective SSM (Mamba-3)
│   └── emotion.py            # EmotionalResponseGenerator
├── tokenizer/         # BPE tokenizer (vocab 32000)
├── agent/             # LLM-powered agent with tools
├── voice/             # STT/TTS voice pipeline
├── browser/           # Headless browser + Tor
└── api/               # FastAPI server
```

## Training Data

Currently **17.4M pre-tokenized tokens** from 4 open-source datasets (OASST1, C4, Alpaca, Dolly). Ready for the first real training run on GPU.

## Status

- [x] Master architecture designed and validated (tiny through 5T configs)
- [x] Text-only forward + loss (validated)
- [x] Text+image forward (validated)
- [x] Text+audio forward (validated)
- [x] Generation (validated)
- [x] Data pipeline with streaming downloads (validated)
- [x] Cognitive Executive (Top Model) integrated
- [x] Reward System (4-dimensional scoring) integrated
- [x] Per-token uncertainty estimation
- [ ] Full training run (needs GPU)
- [ ] Evaluation (MMLU, HumanEval, etc.)
- [ ] Cognitive subsystem ablation studies

## License

All rights reserved. Copyright © NICTOLabs.
