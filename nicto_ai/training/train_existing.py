"""
NICTO Existing Model Training — Full Pipeline
================================================
Trains the original NICTOTrainModel (memory/emotional/creative/reasoning)
with a proper BPE tokenizer and real training data.

This does NOT modify any existing files. It loads the existing model,
adds a BPE tokenizer on top, and trains on real data.

Usage:
    # Step 1: Train tokenizer
    python -m nicto_ai.tokenizer.train --input training_data.jsonl --vocab-size 32000

    # Step 2: Pre-tokenize data
    python -m nicto_ai.training.train_existing --pretokenize --data training_data.jsonl

    # Step 3: Train
    python -m nicto_ai.training.train_existing --steps 2000
"""

import sys
import os
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig


# ============================================================
# BPE TOKENIZER INTEGRATION
# ============================================================

class BPETokenizer:
    """Wrapper to use BPE tokenizer with the existing NICTO model."""

    def __init__(self, tokenizer_path: str = "nicto_ai/tokenizer/artifacts/tokenizer.json"):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        # Map to model's vocab_size by modulo (for compatibility)
        self.model_vocab_size = 32000

    def encode(self, text: str) -> list:
        ids = self.tokenizer.encode(text).ids
        # Modulo mapping to fit model's vocab_size
        return [id % self.model_vocab_size for id in ids]

    def decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids)


# ============================================================
# DATASET WITH BPE
# ============================================================

class BPEJsonlDataset(torch.utils.data.Dataset):
    """JSONL dataset using BPE tokenization."""

    def __init__(self, data_path: str, seq_len: int = 256, tokenizer_path: str = None):
        self.seq_len = seq_len

        # Load tokenizer
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = BPETokenizer(tokenizer_path)
            print(f"Using BPE tokenizer (vocab={self.tokenizer.vocab_size})")
        else:
            self.tokenizer = None
            print("No BPE tokenizer found, using byte-level fallback")

        # Load and tokenize all data
        all_tokens = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if text and len(text) > 50:
                        if self.tokenizer:
                            tokens = self.tokenizer.encode(text)
                        else:
                            tokens = [b % 32000 for b in text.encode("utf-8", errors="ignore")]
                        all_tokens.extend(tokens)
                        all_tokens.append(2)  # EOS separator
                except json.JSONDecodeError:
                    continue

        self.tokens = all_tokens
        self.vocab_size = self.tokenizer.model_vocab_size if self.tokenizer else 32000
        n_samples = max(0, (len(self.tokens) - 1) // seq_len)
        print(f"Dataset: {len(self.tokens):,} tokens, {n_samples:,} samples")

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


# ============================================================
# PRETOKENIZE COMMAND
# ============================================================

def pretokenize_data(data_path: str, tokenizer_path: str, output_path: str):
    """Pre-tokenize JSONL data and save as .bin file."""
    import numpy as np
    tokenizer = BPETokenizer(tokenizer_path)

    all_tokens = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                text = record.get("text", "")
                if text and len(text) > 50:
                    tokens = tokenizer.encode(text)
                    all_tokens.extend(tokens)
                    all_tokens.append(2)  # EOS
            except json.JSONDecodeError:
                continue

    arr = np.array(all_tokens, dtype=np.uint16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_path)
    print(f"Saved {len(all_tokens):,} tokens to {output_path}")


# ============================================================
# LR SCHEDULE
# ============================================================

def get_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# TRAINING
# ============================================================

def train_existing(
    data_path: str = "training_data.jsonl",
    tokenizer_path: str = "nicto_ai/tokenizer/artifacts/tokenizer.json",
    steps: int = 2000,
    batch_size: int = 1,
    grad_accum: int = 4,
    lr: float = 5e-4,
    min_lr: float = 1e-5,
    warmup: int = 50,
    grad_clip: float = 1.0,
    seq_len: int = 256,
    save_every: int = 500,
    eval_every: int = 100,
    resume_from: str = None,
):
    """Train the existing NICTO model with BPE tokenizer."""
    device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"NICTO Existing Model — BPE Training")
    print(f"{'='*60}")

    # Model
    config = NICTOTrainConfig(
        vocab_size=32000, dim=128, max_seq_len=seq_len,
        reasoning_layers=1, n_heads=4, n_kv_heads=2,
        moe_experts=2, moe_activated=1, moe_hidden=128,
        memory_layers=2, emotional_layers=2, creative_layers=2,
    )
    model = NICTOTrainModel(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Resume
    start_step = 0
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # Data
    dataset = BPEJsonlDataset(data_path, seq_len, tokenizer_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Training
    print(f"\nTraining: {start_step} -> {steps} steps")
    print(f"Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"LR: {lr} -> {min_lr}\n")

    model.train()
    step = start_step
    total_loss = 0.0
    start_time = time.time()
    data_iter = iter(dataloader)

    ckpt_dir = Path("checkpoints_v2") / "existing"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, labels=labels)
        loss = output["loss"] / grad_accum
        loss.backward()
        total_loss += output["loss"].item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        current_lr = get_lr(step, warmup, steps, lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        step += 1

        if step % 10 == 0:
            avg = total_loss / 10
            elapsed = time.time() - start_time
            print(f"Step {step:5d}/{steps} | Loss: {avg:.4f} | LR: {current_lr:.2e} | {10/elapsed:.1f} steps/s")
            total_loss = 0.0
            start_time = time.time()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_batch = next(iter(dataloader))
                out = model(eval_batch["input_ids"].to(device), labels=eval_batch["labels"].to(device))
                ppl = math.exp(min(out["loss"].item(), 20))
                print(f"\n  Eval: Loss={out['loss'].item():.4f} PPL={ppl:.2f}")

                # Generate
                prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
                gen = model.generate(prompt, max_new_tokens=30)
                print(f"  Sample: {gen[0].tolist()[:20]}...")
                print()
            model.train()

        if step % save_every == 0:
            path = ckpt_dir / f"step_{step}.pt"
            torch.save({"step": step, "model": model.state_dict(), "loss": output["loss"].item()}, path)
            print(f"  Saved: {path}")

    # Final save
    final_path = ckpt_dir / "final.pt"
    torch.save({"step": step, "model": model.state_dict()}, final_path)
    print(f"\nDone! {step} steps, loss: {output['loss'].item():.4f}")
    print(f"Saved: {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_data.jsonl")
    parser.add_argument("--tokenizer", default="nicto_ai/tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretokenize", action="store_true",
                        help="Pre-tokenize data to .bin file instead of training")
    args = parser.parse_args()

    if args.pretokenize:
        pretokenize_data(args.data, args.tokenizer, "nicto_ai/data/tokenized/data.bin")
    else:
        train_existing(
            data_path=args.data,
            tokenizer_path=args.tokenizer,
            steps=args.steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            seq_len=args.seq_len,
            save_every=args.save_every,
            eval_every=args.eval_every,
            resume_from=args.resume,
        )


if __name__ == "__main__":
    main()
