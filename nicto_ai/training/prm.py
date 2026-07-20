"""
Process Reward Model (PRM) for CSET
====================================
Evaluates each reasoning step, not just the final answer.
Built INTO NICTO's architecture as an extension of the RewardSystem.
"""
import sys, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepScorer(nn.Module):
    """Scores individual reasoning steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, step_repr: torch.Tensor) -> torch.Tensor:
        return self.scorer(step_repr)


class ConsistencyChecker(nn.Module):
    """Checks logical flow between consecutive reasoning steps."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(step_embeddings, step_embeddings, step_embeddings)
        return self.norm(attended)


class StepPredictor(nn.Module):
    """Predicts what the next reasoning step should contain."""

    def __init__(self, dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, step_repr: torch.Tensor) -> torch.Tensor:
        return self.predictor(step_repr)


class ProcessRewardModel(nn.Module):
    """
    Full PRM: scores each step + checks consistency + predicts next step.

    Integrated into NICTO by connecting to the model's hidden states
    at the points where reasoning traces are generated.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.step_scorer = StepScorer(dim)
        self.consistency = ConsistencyChecker(dim, n_heads)
        self.step_predictor = StepPredictor(dim)

    def extract_steps(self, hidden_states: torch.Tensor,
                      step_boundaries: List[Tuple[int, int]]) -> torch.Tensor:
        """Extract mean representation for each reasoning step."""
        step_reprs = []
        for start, end in step_boundaries:
            step_hidden = hidden_states[:, start:end]
            step_repr = step_hidden.mean(dim=1)
            step_reprs.append(step_repr)
        return torch.stack(step_reprs, dim=1)

    def score_steps(self, hidden_states: torch.Tensor,
                    step_boundaries: List[Tuple[int, int]]) -> Dict:
        """Score each step and check consistency."""
        if not step_boundaries:
            return {"step_scores": torch.tensor([]), "consistency": torch.tensor(0.0)}

        step_embeddings = self.extract_steps(hidden_states, step_boundaries)
        step_scores = self.step_scorer(step_embeddings).squeeze(-1)
        consistency = self.consistency(step_embeddings).diagonal(dim1=1, dim2=2).mean()

        return {
            "step_scores": step_scores,
            "consistency": consistency,
            "step_embeddings": step_embeddings,
        }

    def predict_next_step(self, hidden_states: torch.Tensor,
                          last_step_end: int) -> torch.Tensor:
        """Predict the representation of the next reasoning step."""
        if last_step_end >= hidden_states.size(1):
            last_step_end = hidden_states.size(1) - 1
        last_repr = hidden_states[:, last_step_end]
        return self.step_predictor(last_repr)

    def compute_loss(self, hidden_states: torch.Tensor,
                     step_boundaries: List[Tuple[int, int]],
                     targets: torch.Tensor) -> torch.Tensor:
        """
        Compute PRM loss: each step should be scored correctly.

        Args:
            hidden_states: (B, L, dim) from the main model
            step_boundaries: List of (start, end) for each step
            targets: (B, n_steps) target scores (1.0=correct, 0.0=wrong)
        """
        result = self.score_steps(hidden_states, step_boundaries)
        step_scores = result["step_scores"]

        if step_scores.numel() == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        min_len = min(step_scores.size(1), targets.size(1))
        step_scores = step_scores[:, :min_len]
        targets = targets[:, :min_len]

        score_loss = F.binary_cross_entropy(step_scores, targets)

        consistency_penalty = 1.0 - result["consistency"]

        return score_loss + 0.1 * consistency_penalty


class PRMTrainer:
    """Trains the PRM using self-play generated data."""

    def __init__(self, prm: ProcessRewardModel, lr: float = 1e-4):
        self.prm = prm
        self.optimizer = torch.optim.AdamW(prm.parameters(), lr=lr)

    def train_step(self, hidden_states: torch.Tensor,
                   step_boundaries: List[Tuple[int, int]],
                   correct: bool) -> float:
        self.prm.train()
        self.optimizer.zero_grad()

        n_steps = len(step_boundaries)
        if correct:
            targets = torch.ones(1, n_steps, device=hidden_states.device)
        else:
            targets = torch.zeros(1, n_steps, device=hidden_states.device)
            first_wrong = n_steps // 2
            targets[0, :first_wrong] = 1.0

        loss = self.prm.compute_loss(hidden_states, step_boundaries, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.prm.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
