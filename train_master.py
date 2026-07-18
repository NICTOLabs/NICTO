"""
NICTO Master — Training Launcher
==================================
Trains the Master model on pre-tokenized .bin data.
Works on CPU, single GPU, or multi-GPU.

Usage:
  uv run python train_master.py --config tiny --steps 1000
  uv run python train_master.py --config 100m --batch-size 2 --steps 500
"""
import argparse, json, math, os, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nicto_ai.training.model_master import (
    NICTOMasterModel, NICTOMasterConfig,
    config_master_tiny, config_master_100m, config_master_1b, config_master_7b,
)
from nicto_ai.training.data_pipeline_v2 import MMapDataset, MixedMMapDataset, create_dataloader


def get_config(name: str) -> NICTOMasterConfig:
    name = name.lower()
    if name in ("tiny", "t"):
        return config_master_tiny()
    elif name in ("100m", "100"):
        return config_master_100m()
    elif name in ("1b", "1"):
        return config_master_1b()
    elif name in ("7b", "7"):
        return config_master_7b()
    else:
        raise ValueError(f"Unknown config: {name}")


def get_data_sources(data_dir: str = "training_data") -> list:
    """Auto-discover .bin files and create weighted mix."""
    data_dir = Path(data_dir)
    sources = []
    weights_map = {
        "oasst1": 1.0, "dolly": 0.5, "c4": 3.0,
        "alpaca": 0.5, "openorca": 1.0, "cosmopedia": 2.0,
        "fineweb": 3.0, "wikipedia": 2.0, "code": 1.5,
    }
    for f in sorted(data_dir.glob("*.bin")):
        name = f.stem
        weight = weights_map.get(name, 1.0)
        sources.append((str(f), weight))
        print(f"  {name:12s}  weight={weight}")
    if not sources:
        print("  No .bin files found in training_data/. Run prepare_data.py first.")
    return sources


def main():
    parser = argparse.ArgumentParser(description="NICTO Master Training")
    parser.add_argument("--config", type=str, default="tiny", help="Model config: tiny, 100m, 1b, 7b")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps")
    parser.add_argument("--data-dir", type=str, default="training_data", help="Data directory")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--save-every", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\nDevice: {device}")

    # Config
    cfg = get_config(args.config)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    batch_size = getattr(args, "batch_size", 2) or (2 if "tiny" in args.config else 1)
    print(f"Config: {args.config} ({cfg.dim}d, {cfg.n_layers} layers, {cfg.n_heads} heads)")
    print(f"Batch: {batch_size}, Steps: {args.steps}, Seq len: {args.seq_len}")

    # Model
    model = NICTOMasterModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)
    step = 0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resumed from step {step}")

    # Data
    print(f"\nData sources ({args.data_dir}/):")
    sources = get_data_sources(args.data_dir)
    if not sources:
        return
    loader = create_dataloader(sources, seq_len=args.seq_len, batch_size=batch_size)
    data_iter = iter(loader)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training {args.config} for {args.steps} steps...")
    print(f"{'='*60}")
    t0 = time.time()

    for step in range(step, step + args.steps):
        model.train()
        lr = args.lr * min(1.0, (step + 1) / args.warmup) if step < args.warmup else \
             args.lr * 0.5 * (1.0 + math.cos(math.pi * (step - args.warmup) / (args.steps - args.warmup)))

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            elapsed = time.time() - t0
            tps = (step + 1) / max(elapsed, 0.1)
            print(f"  step {step:>6d} | loss {loss.item():>7.4f} | lr {lr:.2e} | "
                  f"{tps:.2f} step/s | unc {out['uncertainty'].mean().item():.3f}")

        if step > 0 and step % args.save_every == 0:
            ckpt_dir = Path(f"checkpoints_master/{args.config}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "loss": loss.item(),
            }, ckpt_dir / f"step_{step}.pt")
            print(f"  ✓ Saved checkpoint at step {step}")

    # Final save
    ckpt_dir = Path(f"checkpoints_master/{args.config}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "loss": loss.item(),
    }, ckpt_dir / "final.pt")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training complete: {args.steps} steps in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Checkpoint: checkpoints_master/{args.config}/final.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
