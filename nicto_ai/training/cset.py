"""
CSET: Cognitive Self-Evolution Training
=======================================
A novel training technique where NICTO teaches itself through:
  Phase 1: THINK  — Chain-of-Thought fundamentals (seed data)
  Phase 2: PLAY   — Self-play problem generation (unlimited data)
  Phase 3: REFLECT — Process Reward Model (step-level judgment)
  Phase 4: EVOLVE  — Evolutionary reasoning optimization
  Phase 5: REMEMBER — Experience buffer (cumulative learning)
"""
import sys, os, time, json, math, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifficultyCalibrator:
    """Dynamically adjusts problem difficulty based on success rate."""

    def __init__(self, initial_difficulty=0.3, window=50, target_success=0.6):
        self.difficulty = initial_difficulty
        self.window = window
        self.target_success = target_success
        self.history = []

    def update(self, success: bool):
        self.history.append(success)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]
        if len(self.history) >= 10:
            rate = sum(self.history) / len(self.history)
            if rate > self.target_success + 0.1:
                self.difficulty = min(1.0, self.difficulty + 0.05)
            elif rate < self.target_success - 0.1:
                self.difficulty = max(0.1, self.difficulty - 0.05)

    def get_level(self) -> int:
        if self.difficulty < 0.3:
            return 1
        elif self.difficulty < 0.5:
            return 2
        elif self.difficulty < 0.7:
            return 3
        elif self.difficulty < 0.9:
            return 4
        else:
            return 5

    def status(self) -> str:
        rate = sum(self.history) / max(len(self.history), 1)
        return f"diff={self.difficulty:.2f} level={self.get_level()} success={rate:.1%}"


class CSETLoss(nn.Module):
    """Combined loss for all CSET phases."""

    def __init__(self, phase: int = 1):
        super().__init__()
        self.phase = phase
        self.weights = {
            "ce": 1.0,
            "reasoning": 0.3,
            "reward": 0.5,
            "diversity": 0.1,
            "prm": 0.2,
            "evolution": 0.1,
            "experience": 0.1,
            "moe_aux": 0.01,
            "mod_entropy": 0.01,
            "exit_entropy": 0.01,
        }

    def forward(self, model_out: Dict, phase_losses: Dict) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(model_out.values())).device
                             if model_out else torch.device("cpu"))

        for key, weight in self.weights.items():
            if key in phase_losses and phase_losses[key] is not None:
                total = total + weight * phase_losses[key]
            elif key == "ce" and "loss" in model_out and model_out["loss"] is not None:
                total = total + weight * model_out["loss"]
            elif key == "moe_aux" and "aux_loss" in model_out:
                total = total + weight * model_out["aux_loss"]
            elif key == "mod_entropy" and "mod_probs" in model_out:
                total = total + weight * self._mod_entropy(model_out["mod_probs"])
            elif key == "exit_entropy" and "exit_entropy" in model_out:
                total = total + weight * model_out["exit_entropy"]

        return total

    def _mod_entropy(self, mod_probs_all):
        if not mod_probs_all:
            return torch.tensor(0.0)
        s = torch.stack([p.mean(dim=0) for p in mod_probs_all], dim=0)
        return -(s * torch.log(s + 1e-8) + (1 - s) * torch.log(1 - s + 1e-8)).mean()


class CSETTrainer:
    """
    Main CSET training orchestrator.
    Manages all 5 phases of Cognitive Self-Evolution Training.
    """

    def __init__(self, model, config, device="cpu"):
        self.model = model
        self.config = config
        self.device = device
        self.calibrator = DifficultyCalibrator()
        self.cset_loss = CSETLoss()
        self.phase = 1
        self.global_step = 0
        self.experience_count = 0

        self.stats = {
            "problems_generated": 0,
            "problems_solved": 0,
            "problems_verified": 0,
            "correct_solutions": 0,
            "evolved_chains": 0,
            "experiences_stored": 0,
        }

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [CSET] {msg}"
        print(line, flush=True)
        with open("cset.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def compute_phase_losses(self, model_out, batch, phase) -> Dict[str, torch.Tensor]:
        losses = {}

        if phase >= 1 and "reasoning_trace" in batch:
            losses["reasoning"] = self._reasoning_loss(model_out, batch)

        if phase >= 2:
            if "reward_loss" in model_out:
                losses["reward"] = model_out["reward_loss"]
            losses["diversity"] = self._diversity_loss(model_out)

        if phase >= 3 and hasattr(self, "prm"):
            losses["prm"] = self._prm_loss(model_out, batch)

        if phase >= 4 and hasattr(self, "evolver"):
            losses["evolution"] = self._evolution_loss(model_out, batch)

        if phase >= 5 and hasattr(self, "experience_buffer"):
            losses["experience"] = self._experience_loss(model_out, batch)

        return losses

    def _reasoning_loss(self, model_out, batch):
        logits = model_out["logits"]
        trace = batch["reasoning_trace"]
        if trace is None or logits.size(1) < 2:
            return torch.tensor(0.0, device=logits.device)
        trace_logits = logits[:, :trace.size(1)]
        trace_labels = trace[:, :trace_logits.size(1)]
        return F.cross_entropy(
            trace_logits.reshape(-1, logits.size(-1)),
            trace_labels.reshape(-1),
            ignore_index=-100,
        )

    def _diversity_loss(self, model_out):
        logits = model_out["logits"]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        return -entropy

    def _prm_loss(self, model_out, batch):
        return torch.tensor(0.0, device=model_out["logits"].device)

    def _evolution_loss(self, model_out, batch):
        return torch.tensor(0.0, device=model_out["logits"].device)

    def _experience_loss(self, model_out, batch):
        return torch.tensor(0.0, device=model_out["logits"].device)

    def step(self, batch, optimizer, phase=None) -> Dict:
        if phase is None:
            phase = self.phase

        self.model.train()
        self.cset_loss.phase = phase

        out = self.model(
            input_ids=batch["input_ids"].to(self.device),
            labels=batch["labels"].to(self.device),
        )

        phase_losses = self.compute_phase_losses(out, batch, phase)
        loss = self.cset_loss(out, phase_losses)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "ce_loss": out.get("loss", torch.tensor(0)).item(),
            "phase": phase,
            "step": self.global_step,
            "difficulty": self.calibrator.difficulty,
            **{f"{k}_loss": v.item() for k, v in phase_losses.items() if v is not None},
        }

    def get_status(self) -> str:
        return (
            f"step={self.global_step} phase={self.phase} "
            f"{self.calibrator.status()} "
            f"generated={self.stats['problems_generated']} "
            f"solved={self.stats['correct_solutions']} "
            f"experiences={self.experience_count}"
        )
