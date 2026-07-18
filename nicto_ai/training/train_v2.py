"""
NICTO Training Script v2 — Production Training Loop
=====================================================
Supports both the existing NICTOTrainModel and the new NICTOModel (v2).

Features:
  - Gradient accumulation + gradient clipping
  - Mixed precision (float16/bfloat16) on GPU, fp32 on CPU
  - Cosine LR schedule with warmup
  - Checkpoint saving/loading with optimizer state
  - W&B logging (optional)
  - Multi-GPU DDP support
  - Text generation samples during training

Usage:
    # Train existing NICTO model with BPE tokenizer
    python -m nicto_ai.training.train_v2 --model existing --config cpu --data data.bin

    # Train new decoder-only model
    python -m nicto_ai.training.train_v2 --model v2 --config 100m --data data.bin
"""

import sys
import os
import time
import math
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# CONFIGS
# ============================================================

TRAIN_CONFIGS = {
    "cpu": {
        "lr": 5e-4, "min_lr": 1e-5, "warmup_steps": 50, "total_steps": 2000,
        "batch_size": 1, "grad_accum": 4, "save_every": 500, "eval_every": 100,
        "grad_clip": 1.0, "weight_decay": 0.1, "seq_len": 256,
        "mixed_precision": False, "num_workers": 0,
    },
    "colab_t4": {
        "lr": 3e-4, "min_lr": 1e-5, "warmup_steps": 100, "total_steps": 5000,
        "batch_size": 4, "grad_accum": 8, "save_every": 500, "eval_every": 250,
        "grad_clip": 1.0, "weight_decay": 0.1, "seq_len": 2048,
        "mixed_precision": "bf16", "num_workers": 2,
    },
    "a100_80g": {
        "lr": 3e-4, "min_lr": 1e-5, "warmup_steps": 200, "total_steps": 20000,
        "batch_size": 8, "grad_accum": 4, "save_every": 1000, "eval_every": 500,
        "grad_clip": 1.0, "weight_decay": 0.1, "seq_len": 4096,
        "mixed_precision": "bf16", "num_workers": 4,
    },
    "a100_8gpus": {
        "lr": 1.5e-4, "min_lr": 1e-5, "warmup_steps": 500, "total_steps": 100000,
        "batch_size": 4, "grad_accum": 2, "save_every": 5000, "eval_every": 1000,
        "grad_clip": 1.0, "weight_decay": 0.1, "seq_len": 4096,
        "mixed_precision": "bf16", "num_workers": 8, "distributed": True,
    },
}


def get_lr(step, warmup_steps, total_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(model_type: str, model_size: str, device: torch.device):
    """
    Create a model instance.

    Args:
        model_type: "existing" for NICTOTrainModel, "v2" for NICTOModel
        model_size: Config name (e.g., "cpu", "100m", "7b")
        device: Target device

    Returns:
        model, model_config
    """
    if model_type == "existing":
        from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig

        size_map = {
            "cpu": NICTOTrainConfig(
                vocab_size=32000, dim=128, max_seq_len=256,
                reasoning_layers=1, n_heads=4, n_kv_heads=2,
                moe_experts=2, moe_activated=1, moe_hidden=128,
                memory_layers=2, emotional_layers=2, creative_layers=2,
            ),
            "100m": NICTOTrainConfig(
                vocab_size=32000, dim=768, max_seq_len=2048,
                reasoning_layers=6, n_heads=12, n_kv_heads=4,
                moe_experts=4, moe_activated=2, moe_hidden=3072,
                memory_layers=4, emotional_layers=4, creative_layers=4,
            ),
        }
        config = size_map.get(model_size, size_map["cpu"])
        model = NICTOTrainModel(config).to(device)
        return model, config

    elif model_type == "v2":
        from nicto_ai.training.model_v2 import (
            NICTOModel, config_100m, config_350m, config_1b, config_7b
        )

        size_map = {
            "100m": config_100m(),
            "350m": config_350m(),
            "1b": config_1b(),
            "7b": config_7b(),
        }
        config = size_map.get(model_size, config_100m())
        model = NICTOModel(config).to(device)
        return model, config

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# TOKENIZER LOADING
# ============================================================

def load_tokenizer(tokenizer_path: str = "nicto_ai/tokenizer/artifacts/tokenizer.json"):
    """Load BPE tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tokenizer_path)


# ============================================================
# TRAINING LOOP
# ============================================================

def train(
    model_type: str = "existing",
    model_size: str = "cpu",
    config_name: str = "cpu",
    data_path: str = None,
    resume_from: str = None,
    tokenizer_path: str = "nicto_ai/tokenizer/artifacts/tokenizer.json",
    use_wandb: bool = False,
):
    """
    Main training function.

    Args:
        model_type: "existing" or "v2"
        model_size: Model size config (cpu, 100m, 350m, 1b, 7b)
        config_name: Training config (cpu, colab_t4, a100_80g, etc.)
        data_path: Path to pre-tokenized .bin file
        resume_from: Checkpoint path to resume from
        tokenizer_path: Path to tokenizer.json
        use_wandb: Enable W&B logging
    """
    # Setup
    train_config = TRAIN_CONFIGS[config_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distributed = train_config.get("distributed", False) and torch.cuda.device_count() > 1

    if distributed:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    print(f"\n{'='*60}")
    print(f"NICTO Training v2")
    print(f"{'='*60}")
    print(f"Model: {model_type} ({model_size})")
    print(f"Config: {config_name}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Data: {data_path or 'synthetic'}")
    print(f"{'='*60}\n")

    # Create model
    model, model_config = create_model(model_type, model_size, device)
    print(f"Parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        betas=(0.9, 0.95),
    )

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        print(f"\nResuming from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"  Could not restore optimizer state: {e}")
        start_step = ckpt.get("step", 0)
        print(f"  Resumed at step {start_step}")

    # Data
    if data_path and Path(data_path).exists():
        from nicto_ai.training.data_pipeline_v2 import create_dataloader
        print(f"Loading data from {data_path}...")
        dataloader = create_dataloader(
            data_path,
            seq_len=train_config["seq_len"],
            batch_size=train_config["batch_size"],
            num_workers=train_config["num_workers"],
            distributed=distributed,
        )
        use_synthetic = False
    else:
        print("No data file found — using synthetic data for pipeline validation")
        dataset = torch.utils.data.TensorDataset(
            torch.randint(0, 32000, (1000, train_config["seq_len"])),
            torch.randint(0, 32000, (1000, train_config["seq_len"])),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=True)
        use_synthetic = True

    # Mixed precision
    use_amp = device.type == "cuda" and train_config.get("mixed_precision")
    amp_dtype = torch.bfloat16 if train_config.get("mixed_precision") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None

    # Checkpoint directory
    ckpt_dir = Path("checkpoints_v2") / model_type / model_size
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if use_wandb:
        try:
            import wandb
            wandb.init(project="nicto-train", config={**train_config, "model": model_type, "size": model_size})
        except ImportError:
            print("W&B not installed, skipping")

    # Training
    print(f"\nStarting training: step {start_step} -> {train_config['total_steps']}")
    print(f"Effective batch: {train_config['batch_size']} x {train_config['grad_accum']} = {train_config['batch_size'] * train_config['grad_accum']}")
    print(f"LR: {train_config['lr']} -> {train_config['min_lr']}\n")

    model.train()
    step = start_step
    total_loss = 0.0
    log_interval = 10
    start_time = time.time()
    data_iter = iter(dataloader)

    while step < train_config["total_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
        labels = batch[1].to(device) if isinstance(batch, (list, tuple)) else batch["labels"].to(device)

        # Forward
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                output = model(input_ids, labels=labels)
                loss = output["loss"] / train_config["grad_accum"]
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            output = model(input_ids, labels=labels)
            loss = output["loss"] / train_config["grad_accum"]
            loss.backward()

        total_loss += output["loss"].item()

        # Gradient accumulation step
        if (step + 1) % train_config["grad_accum"] == 0:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                optimizer.step()
            optimizer.zero_grad()

        # LR schedule
        lr = get_lr(step, train_config["warmup_steps"], train_config["total_steps"],
                     train_config["lr"], train_config["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        step += 1

        # Logging
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            tok_per_sec = (train_config["batch_size"] * train_config["seq_len"] * log_interval) / max(elapsed, 1e-6)
            print(f"Step {step:5d}/{train_config['total_steps']} | Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | {tok_per_sec:.0f} tok/s")
            total_loss = 0.0
            start_time = time.time()

            if use_wandb:
                try:
                    import wandb
                    wandb.log({"loss": avg_loss, "lr": lr, "tok_per_sec": tok_per_sec, "step": step})
                except Exception:
                    pass

        # Eval + sample generation
        if step % train_config["eval_every"] == 0:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 5:
                        break
                    input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
                    labels = batch[1].to(device) if isinstance(batch, (list, tuple)) else batch["labels"].to(device)
                    out = model(input_ids, labels=labels)
                    eval_losses.append(out["loss"].item())
            avg_eval = sum(eval_losses) / len(eval_losses)
            ppl = math.exp(min(avg_eval, 20))
            print(f"\n--- Eval @ Step {step} --- Loss: {avg_eval:.4f} | PPL: {ppl:.2f}")

            # Generate sample
            try:
                prompt = torch.randint(0, model_config.vocab_size if hasattr(model_config, 'vocab_size') else 32000, (1, 10), device=device)
                gen = model.generate(prompt, max_new_tokens=50)
                print(f"  Sample: {gen[0].tolist()[:30]}...")
            except Exception as e:
                print(f"  Gen failed: {e}")
            print()
            model.train()

        # Save checkpoint
        if step % train_config["save_every"] == 0:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": output["loss"].item(),
            }
            path = ckpt_dir / f"step_{step}.pt"
            torch.save(ckpt, path)
            print(f"  Saved: {path}")

    # Final save
    final_ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    final_path = ckpt_dir / "final.pt"
    torch.save(final_ckpt, final_path)
    print(f"\nTraining complete! {step} steps, final loss: {output['loss'].item():.4f}")
    print(f"Saved to: {final_path}")

    if distributed:
        torch.distributed.destroy_process_group()

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NICTO Training v2")
    parser.add_argument("--model", choices=["existing", "v2"], default="existing",
                        help="Model architecture: existing (NICTOTrainModel) or v2 (decoder-only)")
    parser.add_argument("--size", default="cpu", choices=["cpu", "100m", "350m", "1b", "7b"],
                        help="Model size")
    parser.add_argument("--config", default="cpu", choices=list(TRAIN_CONFIGS.keys()),
                        help="Training config")
    parser.add_argument("--data", type=str, default=None, help="Path to pre-tokenized .bin file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--tokenizer", default="nicto_ai/tokenizer/artifacts/tokenizer.json")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    train(
        model_type=args.model,
        model_size=args.size,
        config_name=args.config,
        data_path=args.data,
        resume_from=args.resume,
        tokenizer_path=args.tokenizer,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
