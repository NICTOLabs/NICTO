"""
NICTO Fine-Tuning
==================
Fine-tunes the distillation 3B model on high-quality instruction data
with stricter filtering and higher-quality sources.

Usage:
  python finetune.py --checkpoint checkpoints_master/3b/best.pt --steps 1000
"""
import argparse, math, os, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from nicto_ai.training.model_master import NICTOMasterModel, config_master_3b
from nicto_ai.training.data_pipeline_v2 import create_dataloader

torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"

LOG_FILE = "finetune.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_highquality_sources(data_dir="training_data"):
    """Use only high-quality instruction datasets for fine-tuning."""
    data_dir = Path(data_dir)
    sources = []
    weights = {
        "oasst1": 2.0,
        "alpaca": 2.0,
        "openorca": 2.0,
        "dolly": 1.5,
    }
    for f in sorted(data_dir.glob("*.bin")):
        name = f.stem
        if name in weights:
            sources.append((str(f), weights[name]))
    return sources


def main():
    parser = argparse.ArgumentParser(description="NICTO Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to distillation checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    # Load model
    log(f"Loading checkpoint from {args.checkpoint}...")
    cfg = config_master_3b()
    model = NICTOMasterModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    log(f"Model loaded: {params:,} params, loss={ckpt.get('loss', '?')}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # High-quality data only
    sources = get_highquality_sources()
    log(f"Fine-tuning data: {len(sources)} high-quality sources")
    for s, w in sources:
        log(f"  {Path(s).stem:15s} weight={w}")

    loader = create_dataloader(sources, seq_len=args.seq_len,
                               batch_size=args.batch_size, num_workers=0)
    data_iter = iter(loader)

    ckpt_dir = Path("checkpoints_master/3b")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.steps
    log(f"\n{'='*60}")
    log(f"Fine-tuning for {total_steps} steps")
    log(f"{'='*60}")
    t0 = time.time()
    best_loss = float("inf")
    losses = []

    for step in range(total_steps):
        model.train()

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
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            tps = (step + 1) / max(elapsed, 0.1)
            log(f"step {step:>6d} | loss {avg:>7.4f} | lr {lr:.2e} | {tps:.3f} step/s")

        if step > 0 and step % args.save_every == 0:
            avg = sum(losses[-100:]) / min(len(losses), 100)
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "loss": avg,
            }, ckpt_dir / f"step_ft_{step}.pt")
            if avg < best_loss:
                best_loss = avg
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "loss": avg,
                }, ckpt_dir / "best_ft.pt")
            log(f"  Saved (loss {avg:.4f})")

    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "loss": loss.item(),
    }, ckpt_dir / "final_ft.pt")

    elapsed = time.time() - t0
    log(f"\nFine-tuning complete: {total_steps} steps in {elapsed/60:.1f}m")
    log(f"Final loss: {loss.item():.4f}, Best: {best_loss:.4f}")


if __name__ == "__main__":
    main()
