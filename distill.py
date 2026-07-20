"""
NICTO Knowledge Distillation
=============================
Distills a trained medium teacher (42.8M) into a 3B student model.
Uses soft targets from the teacher to guide the student's learning.

Usage:
  python distill.py --teacher checkpoints_master/medium/best.pt --steps 2000
"""
import argparse, math, os, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from nicto_ai.training.model_master import (
    NICTOMasterModel, NICTOMasterConfig,
    config_master_medium, config_master_200m, config_master_3b,
)
from nicto_ai.training.data_pipeline_v2 import create_dataloader

torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"

LOG_FILE = "distillation.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_data_sources(data_dir="training_data"):
    data_dir = Path(data_dir)
    sources = []
    weights_map = {
        "oasst1": 1.0, "dolly": 0.5, "c4": 3.0,
        "alpaca": 0.5, "openorca": 1.0, "fineweb": 3.0,
        "wikipedia": 2.0,
    }
    for f in sorted(data_dir.glob("*.bin")):
        name = f.stem
        weight = weights_map.get(name, 1.0)
        sources.append((str(f), weight))
    return sources


def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=2.0, alpha=0.5):
    """
    Combined loss:
      - KL divergence on soft targets (teacher -> student)
      - Cross-entropy on hard targets (labels)
    """
    T = temperature

    # Soft targets: KL divergence
    s_logprobs = F.log_softmax(student_logits / T, dim=-1)
    t_probs = F.softmax(teacher_logits / T, dim=-1)
    kl_loss = F.kl_div(
        s_logprobs, t_probs, reduction="batchmean"
    ) * (T * T)

    # Hard targets: cross entropy
    if labels is not None and labels.numel() > 0:
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1), ignore_index=-100,
        )
    else:
        ce_loss = torch.tensor(0.0, device=student_logits.device)

    return alpha * kl_loss + (1 - alpha) * ce_loss


def main():
    parser = argparse.ArgumentParser(description="NICTO Knowledge Distillation")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher checkpoint path")
    parser.add_argument("--config-student", type=str, default="3b", help="Student config")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    log(f"Temperature: {args.temperature}, Alpha: {args.alpha}")

    # Load teacher
    log(f"Loading teacher from {args.teacher}...")
    teacher_cfg = config_master_medium()
    teacher = NICTOMasterModel(teacher_cfg)
    ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters())
    log(f"Teacher loaded: {teacher_params:,} params, loss={ckpt.get('loss', '?')}")

    # Create student
    log(f"Creating student ({args.config_student})...")
    if args.config_student == "3b":
        student_cfg = config_master_3b()
    elif args.config_student in ("200m", "200"):
        student_cfg = config_master_200m()
    elif args.config_student == "medium":
        student_cfg = config_master_medium()
    else:
        raise ValueError(f"Unknown student config: {args.config_student}")

    student = NICTOMasterModel(student_cfg)
    student_params = sum(p.numel() for p in student.parameters())
    log(f"Student created: {student_params:,} params")

    # Try to initialize student from teacher where shapes match
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()
    matched, skipped = 0, 0
    for k in student_sd:
        if k in teacher_sd and student_sd[k].shape == teacher_sd[k].shape:
            student_sd[k] = teacher_sd[k]
            matched += 1
        else:
            skipped += 1
    student.load_state_dict(student_sd)
    log(f"Weight init: {matched} layers copied from teacher, {skipped} random init")

    student.to(device)
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    start_step = 0
    if args.resume:
        rckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(rckpt["model"])
        optimizer.load_state_dict(rckpt["optimizer"])
        start_step = rckpt["step"] + 1
        log(f"Resumed from step {start_step}")

    # Data
    sources = get_data_sources()
    log(f"Data sources: {len(sources)}")
    loader = create_dataloader(sources, seq_len=args.seq_len,
                               batch_size=args.batch_size, num_workers=0)
    data_iter = iter(loader)

    ckpt_dir = Path("checkpoints_master/3b")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    total_steps = start_step + args.steps
    log(f"\n{'='*60}")
    log(f"Distillation: step {start_step} -> {total_steps-1}")
    log(f"{'='*60}")
    t0 = time.time()
    best_loss = float("inf")
    losses = []

    for step in range(start_step, total_steps):
        student.train()

        # LR schedule
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

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(input_ids=input_ids)
            teacher_logits = teacher_out["logits"]

        # Student forward
        student_out = student(input_ids=input_ids)
        student_logits = student_out["logits"]

        # Distillation loss
        # Align logit shapes (teacher/teacher may have different seq_len)
        min_len = min(teacher_logits.size(1), student_logits.size(1))
        loss = distillation_loss(
            student_logits[:, :min_len],
            teacher_logits[:, :min_len],
            labels[:, :min_len] if labels.size(1) >= min_len else labels,
            temperature=args.temperature,
            alpha=args.alpha,
        )

        # Add student's own auxiliary losses
        if student_out["loss"] is not None:
            loss = loss + 0.1 * student_out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            tps = (step - start_step + 1) / max(elapsed, 0.1)
            log(f"step {step:>6d} | loss {avg:>7.4f} | lr {lr:.2e} | {tps:.3f} step/s")

        if step > 0 and step % args.save_every == 0:
            avg = sum(losses[-100:]) / min(len(losses), 100)
            torch.save({
                "step": step,
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": student_cfg,
                "loss": avg,
            }, ckpt_dir / f"step_{step}.pt")
            if avg < best_loss:
                best_loss = avg
                torch.save({
                    "step": step,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": student_cfg,
                    "loss": avg,
                }, ckpt_dir / "best.pt")
            log(f"  Saved checkpoint (loss {avg:.4f})")

    # Final save
    torch.save({
        "step": step,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": student_cfg,
        "loss": loss.item(),
    }, ckpt_dir / "final.pt")

    elapsed = time.time() - t0
    log(f"\nDistillation complete: {args.steps} steps in {elapsed/60:.1f}m")
    log(f"Final loss: {loss.item():.4f}, Best: {best_loss:.4f}")
    log(f"Student: {student_params:,} params")
    log(f"Checkpoint: {ckpt_dir}/final.pt")


if __name__ == "__main__":
    main()
