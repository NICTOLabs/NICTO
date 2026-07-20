# -*- coding: utf-8 -*-
"""
NICTO 3B GPU Training — Google Colab
=====================================
Run this in Google Colab with T4 GPU.

Steps:
  1. Upload this file + the nicto_colab.zip to Colab
  2. Runtime → Change runtime type → T4 GPU
  3. Run all cells
  4. Download checkpoint when done

OR: upload the zip, then run:
  !unzip nicto_colab.zip
  !python colab_train.py
"""
# ─── Cell 1: Setup ───
import subprocess, sys, os, time, json, math, zipfile
from pathlib import Path

# Install torch if needed
try:
    import torch
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "tqdm"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ─── Cell 2: Extract uploaded zip ───
if not Path("nicto_ai").exists():
    # Check for zip
    zips = list(Path(".").glob("nicto_colab*.zip"))
    if zips:
        print(f"Extracting {zips[0]}...")
        with zipfile.ZipFile(zips[0], 'r') as z:
            z.extractall(".")
    else:
        print("ERROR: Upload nicto_colab.zip first!")
        print("  Click Files (left panel) → Upload to session storage")

print("Files:", [p.name for p in Path(".").iterdir() if not p.name.startswith(".")])

# ─── Cell 3: Import NICTO ───
sys.path.insert(0, ".")

from nicto_ai.training.model_master import (
    NICTOMasterModel, NICTOMasterConfig,
    config_master_3b,
)
from nicto_ai.training.data_pipeline_v2 import MMapDataset, MixedMMapDataset, create_dataloader

print("NICTO imported successfully!")

# ─── Cell 4: Create 3B model ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config_master_3b()

model = NICTOMasterModel(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"3B Model: {n_params:,} parameters ({n_params/1e9:.2f}B)")
print(f"VRAM used: {torch.cuda.memory_allocated()/1e6:.0f} MB")

# ─── Cell 5: Setup training ───
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
scaler = GradScaler()  # Mixed precision for 2x speed on T4
start_step = 0

# Check for existing checkpoint (resume training)
ckpt_dir = Path("checkpoints_master/3b")
ckpt_dir.mkdir(parents=True, exist_ok=True)
resume_ckpt = ckpt_dir / "final.pt"
if resume_ckpt.exists():
    ckpt = torch.load(resume_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1
    print(f"Resumed from step {start_step}")

# ─── Cell 6: Load data ───
# Upload training_data/ to Colab or mount Google Drive
data_dir = Path("training_data")
if not data_dir.exists() or not list(data_dir.glob("*.bin")):
    print("Upload training_data/ folder to Colab:")
    print("  Files (left) → training_data/ → upload .bin files")
    print("  OR mount Google Drive and symlink:")
    print("  from google.colab import drive")
    print("  drive.mount('/content/drive')")
    print("  !ln -s /content/drive/MyDrive/NICTO/training_data training_data")
    # Create a tiny dummy file for testing
    data_dir.mkdir(exist_ok=True)
    print("\nWaiting for data upload... (restart after uploading)")

sources = []
weights_map = {
    "oasst1": 1.0, "dolly": 0.5, "c4": 3.0,
    "alpaca": 0.5, "openorca": 1.0, "cosmopedia": 2.0,
    "wikipedia": 2.0, "pile": 1.0, "zen_distill": 1.0,
}
for f in sorted(data_dir.glob("*.bin")):
    if f.stat().st_size > 0:
        w = weights_map.get(f.stem, 1.0)
        sources.append((str(f), w))
        print(f"  {f.stem:12s} {f.stat().st_size/1e6:.1f}MB  weight={w}")

if not sources:
    print("NO DATA FOUND. Upload training .bin files first.")
else:
    loader = create_dataloader(sources, seq_len=1024, batch_size=2, num_workers=2)
    print(f"\nLoaded {len(sources)} datasets")

# ─── Cell 7: Training loop ───
TOTAL_STEPS = 5000
WARMUP = 200
SAVE_EVERY = 500
LOG_EVERY = 25
SEQ_LEN = 1024
BATCH_SIZE = 2

data_iter = iter(loader)
t0 = time.time()
losses = []
best_loss = float("inf")

print(f"\n{'='*60}")
print(f"Training 3B on {device} — {TOTAL_STEPS} steps, seq_len={SEQ_LEN}")
print(f"{'='*60}")

for step in range(start_step, start_step + TOTAL_STEPS):
    model.train()

    # Cosine LR with linear warmup
    if step < WARMUP:
        lr = 3e-4 * (step + 1) / WARMUP
    else:
        ratio = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
        lr = 3e-4 * 0.5 * (1.0 + math.cos(math.pi * ratio))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Mixed precision forward pass
    with autocast():
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]

    # Backward pass with gradient scaling
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    losses.append(loss.item())

    # Log progress
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t0
        avg_loss = sum(losses[-LOG_EVERY:]) / len(losses[-LOG_EVERY:])
        tps = (step - start_step + 1) / max(elapsed, 0.1)
        vram = torch.cuda.memory_allocated() / 1e6
        unc = out["uncertainty"].mean().item() if "uncertainty" in out else 0
        print(f"step {step:>6d} | loss {avg_loss:>7.4f} | lr {lr:.2e} | "
              f"{tps:.2f} step/s | unc {unc:.3f} | vram {vram:.0f}MB")

    # Save checkpoint
    if step > 0 and step % SAVE_EVERY == 0:
        avg = sum(losses[-100:]) / min(len(losses), 100)
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
        print(f"  Checkpoint saved (loss {avg:.4f})")

# Save final
torch.save({
    "step": step,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": cfg,
    "loss": loss.item(),
}, ckpt_dir / "final.pt")

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"Training complete: {TOTAL_STEPS} steps in {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"Final loss: {loss.item():.4f}")
print(f"Best loss: {best_loss:.4f}")
print(f"Checkpoint: checkpoints_master/3b/final.pt")
print(f"{'='*60}")

# ─── Cell 8: Download checkpoint ───
from google.colab import files
print("\nDownloading checkpoint...")
files.download(str(ckpt_dir / "final.pt"))
print("Done! Upload the .pt file to GitHub or your local machine.")
