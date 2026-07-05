"""
NICTO AI - Full Training Script
Works on Colab (T4), local GPU, or k8s cluster.
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig


# ============================================================
# DATASET
# ============================================================
class TextDataset(Dataset):
    def __init__(self, data_path, seq_len=2048, vocab_size=32000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif os.path.isdir(data_path):
            texts = []
            for ext in ["*.txt", "*.md", "*.py", "*.json"]:
                for fp in Path(data_path).rglob(ext):
                    try:
                        texts.append(fp.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        pass
            text = "\n".join(texts)
        else:
            raise FileNotFoundError(f"Data not found: {data_path}")

        # Simple byte-level tokenization
        self.tokens = [b % vocab_size for b in text.encode("utf-8", errors="ignore")]
        print(f"Dataset: {len(self.tokens):,} tokens, {len(self.tokens) // seq_len:,} sequences")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=10000, seq_len=2048, vocab_size=32000):
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
        print(f"Synthetic dataset: {n_samples} samples, {seq_len} seq_len")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx, :-1],
            "labels": self.data[idx, 1:],
        }


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================
def get_lr(step, warmup_steps, total_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# TRAINING LOOP
# ============================================================
def train(config_name="colab"):
    # Config
    if config_name == "colab":
        train_config = {
            "lr": 3e-4, "min_lr": 1e-5, "warmup": 100, "steps": 5000,
            "batch_size": 2, "grad_accum": 8, "save_every": 500,
            "eval_every": 100, "grad_clip": 1.0, "weight_decay": 0.1,
            "seq_len": 1024, "data_path": None,  # None = synthetic
        }
        model_config = NICTOTrainConfig(
            vocab_size=32000, dim=1024, max_seq_len=1024,
            reasoning_layers=6, n_heads=8, n_kv_heads=2,
            moe_experts=4, moe_activated=2, moe_hidden=2048,
            memory_layers=4, emotional_layers=4, creative_layers=4,
        )
    elif config_name == "colab_large":
        train_config = {
            "lr": 3e-4, "min_lr": 1e-5, "warmup": 200, "steps": 10000,
            "batch_size": 1, "grad_accum": 16, "save_every": 1000,
            "eval_every": 200, "grad_clip": 1.0, "weight_decay": 0.1,
            "seq_len": 2048, "data_path": None,
        }
        model_config = NICTOTrainConfig(
            vocab_size=32000, dim=2048, max_seq_len=2048,
            reasoning_layers=8, n_heads=16, n_kv_heads=4,
            moe_experts=8, moe_activated=2, moe_hidden=4096,
            memory_layers=6, emotional_layers=6, creative_layers=6,
        )
    elif config_name == "k8s":
        train_config = {
            "lr": 3e-4, "min_lr": 1e-5, "warmup": 500, "steps": 50000,
            "batch_size": 4, "grad_accum": 4, "save_every": 5000,
            "eval_every": 500, "grad_clip": 1.0, "weight_decay": 0.1,
            "seq_len": 4096, "data_path": None,
        }
        model_config = NICTOTrainConfig(
            vocab_size=32000, dim=4096, max_seq_len=4096,
            reasoning_layers=12, n_heads=32, n_kv_heads=8,
            moe_experts=16, moe_activated=4, moe_hidden=8192,
            memory_layers=8, emotional_layers=8, creative_layers=8,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Model
    print(f"\nCreating model ({config_name})...")
    model = NICTOTrainModel(model_config).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params:,} ({params/1e9:.2f}B)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        betas=(0.9, 0.95),
    )

    # Data
    if train_config["data_path"]:
        dataset = TextDataset(train_config["data_path"], train_config["seq_len"], model_config.vocab_size)
    else:
        print("Using synthetic data (replace with real data for actual training)")
        dataset = SyntheticDataset(n_samples=10000, seq_len=train_config["seq_len"], vocab_size=model_config.vocab_size)

    dataloader = DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=0)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    # Checkpoint dir
    ckpt_dir = Path(__file__).parent / "checkpoints" / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\nStarting training for {train_config['steps']} steps...")
    print(f"Batch size: {train_config['batch_size']} x {train_config['grad_accum']} grad_accum = {train_config['batch_size'] * train_config['grad_accum']} effective")
    print(f"Learning rate: {train_config['lr']} -> {train_config['min_lr']}")
    print()

    model.train()
    step = 0
    epoch = 0
    total_loss = 0
    start_time = time.time()

    while step < train_config["steps"]:
        epoch += 1
        for batch in dataloader:
            if step >= train_config["steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            if use_amp:
                with torch.amp.autocast("cuda"):
                    output = model(input_ids, labels=labels)
                    loss = output["loss"] / train_config["grad_accum"]
                scaler.scale(loss).backward()
            else:
                output = model(input_ids, labels=labels)
                loss = output["loss"] / train_config["grad_accum"]
                loss.backward()

            total_loss += output["loss"].item()

            # Gradient accumulation
            if (step + 1) % train_config["grad_accum"] == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                    optimizer.step()
                optimizer.zero_grad()

            # Learning rate schedule
            lr = get_lr(step, train_config["warmup"], train_config["steps"], train_config["lr"], train_config["min_lr"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            step += 1

            # Logging
            if step % 10 == 0:
                avg_loss = total_loss / 10
                elapsed = time.time() - start_time
                tokens_per_sec = (train_config["batch_size"] * train_config["seq_len"] * 10) / elapsed
                print(f"Step {step:5d}/{train_config['steps']} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s")
                total_loss = 0
                start_time = time.time()

            # Eval
            if step % train_config["eval_every"] == 0:
                model.eval()
                eval_losses = []
                with torch.no_grad():
                    for i, batch in enumerate(dataloader):
                        if i >= 5:
                            break
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        output = model(input_ids, labels=labels)
                        eval_losses.append(output["loss"].item())
                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                perplexity = math.exp(min(avg_eval_loss, 20))
                print(f"\n--- Eval @ Step {step} ---")
                print(f"  Loss: {avg_eval_loss:.4f} | Perplexity: {perplexity:.2f}")
                print(f"  Generating sample...")
                prompt = torch.randint(0, model_config.vocab_size, (1, 10), device=device)
                generated = model.generate(prompt, max_new_tokens=50)
                print(f"  Sample: {generated[0].tolist()[:20]}...")
                print()
                model.train()

            # Save checkpoint
            if step % train_config["save_every"] == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": model_config.__dict__,
                    "train_config": train_config,
                }
                path = ckpt_dir / f"step_{step}.pt"
                torch.save(ckpt, path)
                print(f"  Saved checkpoint: {path}")

    # Final save
    final_ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_config.__dict__,
        "train_config": train_config,
    }
    final_path = ckpt_dir / "final.pt"
    torch.save(final_ckpt, final_path)
    print(f"\nTraining complete! Saved to: {final_path}")
    print(f"Total steps: {step}")
    print(f"Final loss: {avg_loss:.4f}")

    # Save model separately
    model_path = ckpt_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="colab", choices=["colab", "colab_large", "k8s"])
    args = parser.parse_args()
    train(args.config)
