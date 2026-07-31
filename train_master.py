"""
NICTO Master — CPU Training
============================
Optimized for long-running CPU training with auto-resume and progress logging.

Usage:
  python train_master.py --config tiny --steps 10000 --seq-len 512
  python train_master.py --resume checkpoints_master/tiny/step_5000.pt --steps 20000
"""
import argparse, json, math, os, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from nicto_ai.training.model_master import (
    NICTOMasterModel, NICTOMasterConfig,
    config_master_tiny, config_master_medium, config_master_200m,
    config_master_100m, config_master_1b, config_master_7b,
)
from nicto_ai.training.data_pipeline_v2 import MMapDataset, MixedMMapDataset, create_dataloader

# CPU tuning
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

LOG_FILE = "training.log"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_config(name: str) -> NICTOMasterConfig:
    name = name.lower()
    if name in ("tiny", "t"):
        return config_master_tiny()
    elif name in ("medium", "med", "m"):
        return config_master_medium()
    elif name in ("200m", "200"):
        return config_master_200m()
    elif name in ("100m", "100"):
        return config_master_100m()
    elif name in ("1b", "1"):
        return config_master_1b()
    elif name in ("3b", "3"):
        from nicto_ai.training.model_master import config_master_3b
        return config_master_3b()
    elif name in ("7b", "7"):
        return config_master_7b()
    else:
        raise ValueError(f"Unknown config: {name}")


def get_data_sources(data_dir: str = "training_data") -> list:
    data_dir = Path(data_dir)
    sources = []
    weights_map = {
        "oasst1": 1.0, "dolly": 0.5, "c4": 3.0,
        "alpaca": 0.5, "openorca": 1.0, "cosmopedia": 2.0,
        "fineweb": 3.0, "wikipedia": 2.0, "code": 1.5,
    }
    for f in sorted(data_dir.glob("*.bin")):
        name = f.stem
        if f.stat().st_size == 0:
            continue
        weight = weights_map.get(name, 1.0)
        sources.append((str(f), weight))
        log(f"  Data: {name:12s}  weight={weight}")
    if not sources:
        log("  No .bin files found in training_data/. Run prepare_data.py first.")
    return sources


def main():
    parser = argparse.ArgumentParser(description="NICTO Master CPU Training")
    parser.add_argument("--config", type=str, default="tiny", help="Model config: tiny, 100m, 1b, 7b")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--data-dir", type=str, default="training_data", help="Data directory")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--log-every", type=int, default=50, help="Log interval")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    # Auto-detect best device
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        log("Device: TPU (XLA)")
    except ImportError:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Device: {device}")

    cfg = get_config(args.config)
    log(f"Device: {device}")
    log(f"Config: {args.config} ({cfg.dim}d, {cfg.n_layers} layers, {cfg.n_heads} heads)")
    log(f"Params: {sum(p.numel() for p in NICTOMasterModel(cfg).parameters()):,}")
    log(f"Batch: {args.batch_size}, Steps: {args.steps}, Seq len: {args.seq_len}")

    model = NICTOMasterModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    start_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", -1) + 1
        log(f"Resumed from step {ckpt.get('step', 0)} (loss {float(ckpt.get('loss', 0.0)):.4f})")

    log(f"\nData sources ({args.data_dir}/):")
    sources = get_data_sources(args.data_dir)
    if not sources:
        return
    loader = create_dataloader(sources, seq_len=args.seq_len, batch_size=args.batch_size, num_workers=0)
    data_iter = iter(loader)

    # Training loop
    total_steps = start_step + args.steps
    log(f"\n{'='*60}")
    log(f"Training {args.config} from step {start_step} to {total_steps-1}...")
    log(f"{'='*60}")
    t0 = time.time()
    losses = []
    best_loss = float("inf")

    for step in range(start_step, total_steps):
        model.train()

        # Cosine LR with linear warmup
        if step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        else:
            ratio = (step - args.warmup) / (total_steps - args.warmup)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * ratio))
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

        losses.append(loss.item())

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            tps = (step - start_step + 1) / max(elapsed, 0.1)
            log(f"step {step:>6d} | loss {avg_loss:>7.4f} | lr {lr:.2e} | "
                f"{tps:.2f} step/s | unc {out['uncertainty'].mean().item():.3f}")

        if step > 0 and step % args.save_every == 0:
            avg = sum(losses[-100:]) / min(len(losses), 100)
            ckpt_dir = Path(f"checkpoints_master/{args.config}")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "loss": avg,
            }, ckpt_dir / f"step_{step}.pt")
            if avg < best_loss:
                best_loss = avg
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "loss": avg,
                }, ckpt_dir / "best.pt")
            log(f"  [saved] Checkpoint (loss {avg:.4f})")

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
    log(f"\n{'='*60}")
    log(f"Training complete: {args.steps} steps in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log(f"Final loss: {loss.item():.4f}")
    log(f"Best loss: {best_loss:.4f}")
    log(f"Checkpoint: checkpoints_master/{args.config}/final.pt")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
