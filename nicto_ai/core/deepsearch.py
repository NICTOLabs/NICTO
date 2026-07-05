"""
NICTO AI - DeepSearch Module
Chain-of-thought reasoning with iterative deepening and self-evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchState:
    """State of a reasoning chain during search"""
    hidden: torch.Tensor
    score: float
    steps_taken: int
    tokens: List[int]
    is_finished: bool = False


class ThoughtGenerator(nn.Module):
    """Generates candidate reasoning steps"""

    def __init__(self, dim: int = 8192, n_heads: int = 16, max_thoughts: int = 8):
        super().__init__()
        self.dim = dim
        self.max_thoughts = max_thoughts

        # Think step projector
        self.think_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Multi-head thought generation (divergent thinking)
        self.thought_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, dim),
            ) for _ in range(max_thoughts)
        ])

        # Thought selection gate
        self.selection_gate = nn.Sequential(
            nn.Linear(dim * max_thoughts, max_thoughts),
            nn.Softmax(dim=-1),
        )

        # Thought diversity regularizer
        self.diversity_proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden: torch.Tensor,
        n_thoughts: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple candidate reasoning steps.

        Args:
            hidden: Current hidden state [batch, dim]
            n_thoughts: Number of thoughts to generate

        Returns:
            thoughts: [batch, n_thoughts, dim]
            weights: [batch, n_thoughts]
        """
        n_thoughts = min(n_thoughts, self.max_thoughts)
        projected = self.think_proj(hidden)

        # Generate diverse thoughts
        thoughts = []
        for i in range(n_thoughts):
            thought = self.thought_heads[i](projected)
            thoughts.append(thought)
        thoughts = torch.stack(thoughts, dim=1)  # [B, n_thoughts, dim]

        # Score and select
        flat_thoughts = thoughts.reshape(hidden.shape[0], -1)
        n_pad = self.max_thoughts - n_thoughts
        if n_pad > 0:
            pad = torch.zeros(flat_thoughts.shape[0], n_pad * self.dim, device=hidden.device)
            flat_thoughts = torch.cat([flat_thoughts, pad], dim=-1)

        weights = self.selection_gate(flat_thoughts)[:, :n_thoughts]

        # Diversity bonus via orthogonal projection
        div_proj = self.diversity_proj(thoughts)
        sim_matrix = torch.bmm(div_proj, div_proj.transpose(1, 2))
        eye = torch.eye(n_thoughts, device=hidden.device).unsqueeze(0)
        diversity_loss = (sim_matrix * eye).sum(dim=-1).mean()

        return thoughts, weights, diversity_loss


class SearchEvaluator(nn.Module):
    """Evaluates quality of reasoning states"""

    def __init__(self, dim: int = 8192):
        super().__init__()
        self.dim = dim

        # Value head (how promising is this state?)
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Progress detector (are we making progress?)
        self.progress_detector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        # Completion classifier
        self.completion_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 3),  # not_ready, ready, exhausted
        )

    def forward(
        self,
        current: torch.Tensor,
        initial: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate reasoning state.

        Args:
            current: Current hidden state [batch, dim]
            initial: Initial hidden state [batch, dim] for progress comparison

        Returns:
            Dictionary with value, progress, completion logits
        """
        value = self.value_head(current).squeeze(-1)

        if initial is not None:
            progress_input = torch.cat([current, initial], dim=-1)
        else:
            progress_input = torch.cat([current, current], dim=-1)
        progress = self.progress_detector(progress_input).squeeze(-1)

        completion = self.completion_head(current)

        return {
            "value": value,
            "progress": progress,
            "completion_logits": completion,
        }


class DeepSearchModule(nn.Module):
    """
    DeepSearch - Iterative deepening reasoning engine

    Implements a beam search over reasoning chains:
    1. Generate multiple candidate thoughts at each step
    2. Evaluate each candidate's promise
    3. Keep top-k candidates (beam search)
    4. Continue until completion signal or max depth

    Inspired by AlphaGo's MCTS + chain-of-thought prompting.
    """

    def __init__(
        self,
        dim: int = 8192,
        n_heads: int = 16,
        max_depth: int = 10,
        beam_width: int = 4,
        n_thoughts_per_step: int = 4,
        completion_threshold: float = 0.8,
    ):
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.n_thoughts = n_thoughts_per_step
        self.completion_threshold = completion_threshold

        # Core components
        self.thought_generator = ThoughtGenerator(dim, n_heads, max_thoughts=8)
        self.evaluator = SearchEvaluator(dim)

        # State projection (for maintaining state across steps)
        self.state_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # Step embedding (tells the model which step we're on)
        self.step_embedding = nn.Embedding(max_depth, dim)

        # Output merger (combines best reasoning chain)
        self.output_merger = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def beam_search(
        self,
        initial_hidden: torch.Tensor,
        n_thoughts: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Run beam search reasoning.

        Args:
            initial_hidden: Starting state [batch, dim]
            n_thoughts: Thoughts per step

        Returns:
            Dictionary with best_reasoning, all_scores, depth_reached
        """
        batch_size = initial_hidden.shape[0]
        device = initial_hidden.device

        # Initialize beam with initial state
        initial_eval = self.evaluator(initial_hidden, initial_hidden)
        beams = [{
            "hidden": initial_hidden,
            "score": initial_eval["value"],
            "depth": 0,
            "accumulated": initial_hidden,
        }]

        all_scores = []
        best_final = None
        best_score = torch.tensor(-float("inf"), device=device)

        for depth in range(self.max_depth):
            step_emb = self.step_embedding(
                torch.tensor(depth, device=device)
            ).unsqueeze(0).expand(batch_size, -1)

            candidates = []
            for beam in beams:
                if beam["score"].min() < 0:
                    continue

                state = beam["hidden"] + step_emb

                # Generate thoughts
                thoughts, weights, div_loss = self.thought_generator(state, n_thoughts)

                # Evaluate each thought
                for i in range(n_thoughts):
                    thought = thoughts[:, i]
                    eval_result = self.evaluator(thought, initial_hidden)

                    # Score = value + progress bonus - depth penalty
                    depth_penalty = 0.02 * depth
                    score = eval_result["value"] + 0.5 * eval_result["progress"] - depth_penalty

                    # Combine with parent score
                    combined_score = beam["score"] + score * weights[:, i]

                    candidate = {
                        "hidden": self.state_proj(thought),
                        "score": combined_score,
                        "depth": depth + 1,
                        "accumulated": self.output_merger(
                            torch.cat([beam["accumulated"], thought], dim=-1)
                        ),
                    }
                    candidates.append(candidate)

                    # Check completion
                    completion_probs = F.softmax(eval_result["completion_logits"], dim=-1)
                    if completion_probs[:, 1].mean() > self.completion_threshold:
                        if combined_score.min() > best_score.min():
                            best_final = candidate
                            best_score = combined_score

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda c: c["score"].mean().item(), reverse=True)
            beams = candidates[:self.beam_width]

            all_scores.append({
                "depth": depth,
                "best_score": beams[0]["score"].mean().item(),
                "n_candidates": len(candidates),
            })

            # Early stop if all beams completed
            if best_final is not None and depth >= 2:
                break

        # Return best result
        if best_final is None:
            best_final = beams[0] if beams else {
                "hidden": initial_hidden,
                "accumulated": initial_hidden,
                "score": torch.zeros(batch_size, device=device),
                "depth": 0,
            }

        return {
            "best_reasoning": best_final["accumulated"],
            "best_hidden": best_final["hidden"],
            "best_score": best_final["score"],
            "depth_reached": best_final["depth"],
            "all_scores": all_scores,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        use_search: bool = True,
        n_thoughts: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Process through DeepSearch.

        Args:
            hidden: Input hidden state [batch, seq_len, dim] or [batch, dim]
            use_search: Whether to use beam search or single pass
            n_thoughts: Thoughts per step (only used with search)

        Returns:
            Dictionary with output, scores, depth
        """
        if hidden.dim() == 3:
            # Pool to [batch, dim]
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        if use_search and self.training is False:
            # Beam search at inference
            result = self.beam_search(pooled, n_thoughts)
        else:
            # Single-step during training (differentiable)
            thoughts, weights, div_loss = self.thought_generator(pooled, n_thoughts)

            # Weighted combination
            weighted_thought = (thoughts * weights.unsqueeze(-1)).sum(dim=1)

            eval_result = self.evaluator(weighted_thought, pooled)

            result = {
                "best_reasoning": self.output_merger(
                    torch.cat([pooled, weighted_thought], dim=-1)
                ),
                "best_hidden": weighted_thought,
                "best_score": eval_result["value"],
                "depth_reached": 1,
                "diversity_loss": div_loss,
            }

        return result
