"""
NICTO AI - Full Training Script
Works on Colab (T4), local GPU, or k8s cluster.
Supports both synthetic and real data.
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
from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig, nicto_5b_config


# ============================================================
# DATA DIRECTORY
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# DATASET
# ============================================================
class TextDataset(Dataset):
    """Load text from files (txt, md, py, json, jsonl)."""
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


class JsonlDataset(Dataset):
    """Load text from JSONL files (one JSON object per line with 'text' field)."""
    def __init__(self, data_path, seq_len=2048, vocab_size=32000, max_samples=None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        texts = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            jsonl_files = [data_path]
        elif data_path.is_dir():
            jsonl_files = list(data_path.glob("*.jsonl"))
        else:
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        for fp in jsonl_files:
            print(f"Loading {fp}...")
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if max_samples and len(texts) >= max_samples:
                        break
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if text and len(text) > 100:
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
        
        # Concatenate all texts with separator
        full_text = "\n\n".join(texts)
        self.tokens = [b % vocab_size for b in full_text.encode("utf-8", errors="ignore")]
        
        print(f"JsonlDataset: {len(texts):,} documents, {len(self.tokens):,} tokens, {len(self.tokens) // seq_len:,} sequences")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


class MixedDataset(Dataset):
    """Mix multiple datasets with weighted sampling."""
    def __init__(self, dataset_configs, seq_len=2048, vocab_size=32000):
        """
        Args:
            dataset_configs: list of (path, weight) tuples
            seq_len: sequence length
            vocab_size: vocabulary size
        """
        self.seq_len = seq_len
        self.datasets = []
        self.weights = []
        self.cumulative_sizes = []
        
        total_weight = sum(w for _, w in dataset_configs)
        
        for path, weight in dataset_configs:
            try:
                ds = JsonlDataset(path, seq_len, vocab_size)
                self.datasets.append(ds)
                self.weights.append(weight / total_weight)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        # Calculate cumulative sizes for weighted sampling
        cumulative = 0
        for ds, w in zip(self.datasets, self.weights):
            cumulative += len(ds) * w
            self.cumulative_sizes.append(cumulative)
        
        self.total_size = int(cumulative)
        print(f"MixedDataset: {len(self.datasets)} datasets, {self.total_size:,} effective samples")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Find which dataset to sample from based on weight
        import random
        r = random.random()
        for i, (cum_size, w) in enumerate(zip(self.cumulative_sizes, self.weights)):
            if r < cum_size / self.total_size:
                ds_idx = i
                break
        else:
            ds_idx = len(self.datasets) - 1
        
        # Sample from selected dataset
        sample_idx = random.randint(0, len(self.datasets[ds_idx]) - 1)
        return self.datasets[ds_idx][sample_idx]


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
def train(config_name="colab", data_path=None, data_mix=None, resume_from=None):
    # Config
    if config_name == "colab":
        train_config = {
            "lr": 3e-4, "min_lr": 1e-5, "warmup": 100, "steps": 5000,
            "batch_size": 2, "grad_accum": 8, "save_every": 500,
            "eval_every": 100, "grad_clip": 1.0, "weight_decay": 0.1,
            "seq_len": 1024, "data_path": None,  # None = synthetic
            "data_mix": None,  # None = single dataset or synthetic
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
            "data_mix": None,
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
            "data_mix": None,
        }
        model_config = NICTOTrainConfig(
            vocab_size=32000, dim=4096, max_seq_len=4096,
            reasoning_layers=12, n_heads=32, n_kv_heads=8,
            moe_experts=16, moe_activated=4, moe_hidden=8192,
            memory_layers=8, emotional_layers=8, creative_layers=8,
        )
    elif config_name == "5b":
        # ~5B params. Needs an A100/H100-class GPU (80GB) for full fine-tuning,
        # or a 24GB+ GPU using LoRA/QLoRA. Will NOT fit on a free T4 (16GB).
        train_config = {
            "lr": 1.5e-4, "min_lr": 1e-5, "warmup": 1000, "steps": 20000,
            "batch_size": 1, "grad_accum": 32, "save_every": 1000,
            "eval_every": 200, "grad_clip": 1.0, "weight_decay": 0.1,
            "seq_len": 4096, "data_path": None,
            "data_mix": None,
        }
        model_config = nicto_5b_config()
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Apply CLI overrides (previously parsed but silently discarded)
    if data_path:
        train_config["data_path"] = data_path
    if data_mix:
        # data_mix passed as list of "path:weight" strings -> list of (path, weight) tuples
        parsed_mix = []
        for item in data_mix:
            path, weight = item.rsplit(":", 1)
            parsed_mix.append((path, float(weight)))
        train_config["data_mix"] = parsed_mix

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

    # Resume from checkpoint (warm start) if provided
    start_step = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(
                f"  Warning: {len(missing)} missing keys (architecture mismatch) - those params stay randomly initialized"
            )
        if unexpected:
            print(
                f"  Warning: {len(unexpected)} unexpected keys in checkpoint were ignored"
            )
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(
                    f"  Could not restore optimizer state ({e}); starting optimizer fresh"
                )
        start_step = ckpt.get("step", 0)
        print(f"  Resumed at step {start_step}")

    # Data loading
    if train_config.get("data_mix"):
        # Load multiple datasets with mixing
        print(f"Loading mixed datasets...")
        dataset_configs = []
        for path, weight in train_config["data_mix"]:
            full_path = PROCESSED_DIR / path if not os.path.isabs(path) else Path(path)
            dataset_configs.append((str(full_path), weight))
        dataset = MixedDataset(dataset_configs, train_config["seq_len"], model_config.vocab_size)
    elif train_config["data_path"]:
        data_path = train_config["data_path"]
        if data_path.endswith(".jsonl") or (os.path.isdir(data_path) and any(Path(data_path).glob("*.jsonl"))):
            print(f"Loading JSONL data from {data_path}...")
            dataset = JsonlDataset(data_path, train_config["seq_len"], model_config.vocab_size)
        else:
            print(f"Loading text data from {data_path}...")
            dataset = TextDataset(data_path, train_config["seq_len"], model_config.vocab_size)
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
    step = start_step
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
    parser.add_argument("--config", default="colab", choices=["colab", "colab_large", "k8s", "5b"])
    parser.add_argument("--data-path", type=str, default=None, help="Path to training data (file or directory)")
    parser.add_argument("--data-mix", type=str, nargs="+", help="Mixed datasets: path1:weight1 path2:weight2 ...")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to warm-start from")
    parser.add_argument("--use-real-data", action="store_true",
        help="Shortcut: use the README's recommended Priority-1 mix from nicto_ai/data/processed/ "
        "(requires running `python -m nicto_ai.data.collect --priority 1 --process` first)")
    args = parser.parse_args()

    data_mix = args.data_mix
    if args.use_real_data and not data_mix:
        # Matches the README's "Recommended Training Mix (10B tokens)" table.
        # Expects processed JSONL files at nicto_ai/data/processed/<name>.jsonl
        # (produced by: python -m nicto_ai.data.collect --priority 1 --process)
        data_mix = [
            "fineweb-edu.jsonl:0.30",
            "slimpajama.jsonl:0.20",
            "wikipedia.jsonl:0.10",
            "the_stack_python.jsonl:0.10",
            "openhermes_2.5.jsonl:0.10",
            "math.jsonl:0.10",
            "ultrachat.jsonl:0.10",
        ]
        print(
            "Using README's recommended Priority-1 data mix (see nicto_ai/data/collect.py to fetch these files first)"
        )

    if args.data_path:
        print(f"Using data path: {args.data_path}")
    if data_mix:
        print(f"Using data mix: {data_mix}")
    if args.resume:
        print(f"Will resume from: {args.resume}")

    train(
        args.config,
        data_path=args.data_path,
        data_mix=data_mix,
        resume_from=args.resume,
    )
