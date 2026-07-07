# NICTO AI

Open-source neural architecture — reasoning, memory, emotion, and consciousness in one model.

**Status:** Undergoing training on Google Colab (T4 GPU). Current checkpoint is ~207M parameters. No benchmarks run yet.

## Quick start

```bash
git clone https://github.com/NICTOLabs/NICTO.git
cd NICTO
pip install -e .
```

### Test the model

```bash
python -c "
import torch
from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig

config = NICTOTrainConfig()
model = NICTOTrainModel(config)
model.load_state_dict(torch.load('nicto_model_final.pt', weights_only=True))

x = torch.randint(0, config.vocab_size, (1, 10))
out = model.generate(x, max_new_tokens=50)
print('Generated tokens:', out.shape[1] - 10)
"
```

### Run the voice engine

```python
from nicto_ai.voice.backend_interface import NICTOBackend, Message

backend = NICTOBackend(checkpoint_path="nicto_model_final.pt")
result = backend.complete([Message(role="user", content="Hello")], max_tokens=50)
print(result.text)
```

### Train

```bash
# Quick smoke test (synthetic data)
python -m nicto_ai.training.train --config colab

# With real data
python -m nicto_ai.training.train --config colab --use-real-data

# Download training datasets first
python -m nicto_ai.data.collect --priority 1 --process
```

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
- ~2GB disk for the checkpoint (Git LFS)

## How it's trained

NICTO is trained on Google Colab Free (T4 16GB) using PyTorch. The training pipeline supports:
- Synthetic data (for quick testing)
- Real datasets from HuggingFace (Wikipedia, code, reasoning)
- Checkpoint save/resume

## License

Apache-2.0
