"""
NICTO CSET Training — Cognitive Self-Evolution Training
=======================================================
Usage:
  python train_cset.py --config medium --phase 1 --steps 5000
  python train_cset.py --config 200m --phase 2 --steps 10000
  python train_cset.py --config medium --resume checkpoints_cset/phase1/best.pt --phase 2
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
    config_master_3b,
)
from nicto_ai.training.data_pipeline_v2 import MMapDataset, MixedMMapDataset, create_dataloader
from nicto_ai.training.cset import CSETTrainer, DifficultyCalibrator
from nicto_ai.training.prm import ProcessRewardModel, PRMTrainer
from nicto_ai.training.reasoning_evolver import ReasoningEvolver
from nicto_ai.training.experience_buffer import ExperienceBuffer, Experience
from nicto_ai.training.propose_verify import SelfPlayLoop
from nicto_ai.training.dcot_tags import DCoTPrompter, DifficultyScaler

torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

LOG_FILE = "cset_training.log"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_config(name: str) -> NICTOMasterConfig:
    name = name.lower()
    configs = {
        "tiny": config_master_tiny, "t": config_master_tiny,
        "medium": config_master_medium, "med": config_master_medium, "m": config_master_medium,
        "200m": config_master_200m, "200": config_master_200m,
        "3b": config_master_3b, "3": config_master_3b,
    }
    if name in configs:
        return configs[name]()
    raise ValueError(f"Unknown config: {name}")


def get_data_sources(data_dir: str = "training_data") -> list:
    data_dir = Path(data_dir)
    sources = []
    weights_map = {
        "oasst1": 1.0, "dolly": 0.5, "c4": 3.0,
        "alpaca": 0.5, "openorca": 1.0, "wikipedia": 2.0,
        "pile": 1.0, "zen_distill": 1.0,
    }
    for f in sorted(data_dir.glob("*.bin")):
        if f.stat().st_size > 0:
            weight = weights_map.get(f.stem, 1.0)
            sources.append((str(f), weight))
            log(f"  Data: {f.stem:12s}  weight={weight}")
    return sources


def main():
    parser = argparse.ArgumentParser(description="NICTO CSET Training")
    parser.add_argument("--config", type=str, default="medium", help="Model config")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="CSET phase to train")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--save-every", type=int, default=500, help="Save interval")
    parser.add_argument("--log-every", type=int, default=25, help="Log interval")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--data-dir", type=str, default="training_data")
    parser.add_argument("--experience-buffer", type=str, default="cset_experiences.json")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    cfg = get_config(args.config)
    n_params = sum(p.numel() for p in NICTOMasterModel(cfg).parameters())
    log(f"Config: {args.config} ({cfg.dim}d, {cfg.n_layers} layers, {cfg.n_heads} heads)")
    log(f"Params: {n_params:,} ({n_params/1e6:.1f}M)")
    log(f"Phase: {args.phase}, Steps: {args.steps}, Seq len: {args.seq_len}")

    model = NICTOMasterModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    start_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0) + 1
        log(f"Resumed from step {start_step}")

    cset = CSETTrainer(model, cfg, device)
    cset.phase = args.phase

    if args.phase >= 3:
        prm = ProcessRewardModel(cfg.dim).to(device)
        cset.prm = prm
        cset.prm_trainer = PRMTrainer(prm)

    if args.phase >= 4:
        cset.evolver = ReasoningEvolver()

    cset.experience_buffer = ExperienceBuffer()
    exp_path = Path(args.experience_buffer)
    if exp_path.exists():
        cset.experience_buffer.load(str(exp_path))
        log(f"Loaded {cset.experience_buffer.stats()['total']} experiences")

    self_play = SelfPlayLoop()
    dcot = DCoTPrompter()
    diff_scaler = DifficultyScaler()

    sources = get_data_sources(args.data_dir)
    if not sources:
        log("No training data found!")
        return
    loader = create_dataloader(sources, seq_len=args.seq_len, batch_size=args.batch_size, num_workers=0)
    data_iter = iter(loader)

    ckpt_dir = Path(f"checkpoints_cset/phase{args.phase}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = start_step + args.steps
    log(f"\n{'='*60}")
    log(f"CSET Phase {args.phase}: Training from step {start_step} to {total_steps-1}")
    log(f"{'='*60}")

    t0 = time.time()
    losses = []
    best_loss = float("inf")

    for step in range(start_step, total_steps):
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

        if args.phase >= 2:
            diff_scaler.advance()
            difficulty = diff_scaler.get_difficulty()
            problem = self_play.generate_problem(difficulty)
            dcot_text = dcot.add_tags(
                "", problem["domain"], difficulty
            )
            batch_info = {
                "domain": problem["domain"],
                "difficulty": difficulty,
                "dcot_tags": dcot_text,
            }
        else:
            batch_info = {}

        result = cset.step(batch, optimizer)

        losses.append(result["loss"])

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            tps = (step - start_step + 1) / max(elapsed, 0.1)
            log(f"step {step:>6d} | loss {avg_loss:>7.4f} | lr {lr:.2e} | "
                f"{tps:.2f} step/s | phase {args.phase} | {cset.get_status()}")

        if step > 0 and step % args.save_every == 0:
            avg = sum(losses[-100:]) / min(len(losses), 100)
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "loss": avg,
                "phase": args.phase,
                "cset_stats": cset.stats,
            }, ckpt_dir / f"step_{step}.pt")
            if avg < best_loss:
                best_loss = avg
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                    "loss": avg,
                    "phase": args.phase,
                    "cset_stats": cset.stats,
                }, ckpt_dir / "best.pt")
            log(f"  Checkpoint saved (loss {avg:.4f})")

    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "loss": loss.item() if 'loss' in dir() else losses[-1],
        "phase": args.phase,
        "cset_stats": cset.stats,
    }, ckpt_dir / "final.pt")

    cset.experience_buffer.save(str(exp_path))

    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"CSET Phase {args.phase} complete: {args.steps} steps in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log(f"Best loss: {best_loss:.4f}")
    log(f"Checkpoint: {ckpt_dir / 'final.pt'}")
    log(f"Stats: {cset.get_status()}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
