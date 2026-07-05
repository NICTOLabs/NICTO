# NICTO Training Prompt for Gemini (Colab)

Use this prompt in Google Colab with Gemini to train the NICTO model.

---

## PROMPT:

```
I have the NICTO AI project on GitHub: https://github.com/NICTOLabs/-NICTO

The project contains a neural network architecture with these components:
- MLA (Multi-Latent Attention) - efficient long-context processing
- MoE (Mixture of Experts) - sparse computation, top-2 routing
- Mamba SSM - state-space model for sequence modeling
- Liquid Neural Networks - continuous-time dynamics
- Consciousness layer - metacognition, uncertainty estimation
- Emotion system -情感状态输出
- Hierarchical memory - working/episodic/semantic
- Fusion gate - 4-way output fusion (Reasoning, Memory, Emotional, Creative)

The trainable model is in: nicto_ai/training/model_train.py (NICTOTrainModel class)

TASK: Write a complete Google Colab notebook that:

1. Clones the NICTO repo from GitHub
2. Installs dependencies (torch, transformers, datasets)
3. Loads a training dataset (use RedPajama-V2 or SlimPajama sample)
4. Creates the NICTOTrainModel with ~100M params (fits on T4 16GB)
5. Trains for 5000 steps with:
   - Mixed precision (bf16)
   - Gradient checkpointing
   - Cosine LR schedule with warmup
   - Gradient accumulation (effective batch ~32)
6. Saves checkpoints every 500 steps
7. Generates sample text every 500 steps to show progress
8. Plots training loss curve
9. Saves the final model

CONSTRAINTS:
- Must run on Google Colab Free tier (T4 16GB, ~12hr max)
- Model must fit in VRAM with batch_size >= 2
- Use the existing NICTOTrainModel class from the repo
- No mocking - real training only
- Track and report: final loss, perplexity, tokens/second

The NICTOTrainModel config for T4:
- vocab_size=32000
- dim=1024  
- max_seq_len=1024
- reasoning_layers=6
- n_heads=8, n_kv_heads=2
- moe_experts=4, moe_activated=2
- memory_layers=4, emotional_layers=4, creative_layers=4

Please generate the complete, runnable Colab notebook.
```

---

## ALTERNATIVE SHORT PROMPT:

```
Clone https://github.com/NICTOLabs/-NICTO and train the NICTOTrainModel from nicto_ai/training/model_train.py on a text dataset using Google Colab T4 GPU. Use mixed precision, gradient checkpointing, and save checkpoints. Show training loss and generate samples during training. Model config: dim=1024, 6 reasoning layers, 4 MoE experts. Train for 5000 steps.
```
