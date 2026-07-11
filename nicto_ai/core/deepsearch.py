"""
NICTO AI - DeepSearch Module v2

Chain-of-thought reasoning with:
  - Iterative deepening beam search
  - Monte Carlo Tree Search (MCTS) style exploration
  - Reflexion (self-reflection and learning from mistakes)
  - Multi-step planning with sub-goal decomposition
  - Uncertainty-aware exploration
  - Self-evaluation and progress tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class SearchState:
    """State of a reasoning chain during search"""
    hidden: torch.Tensor
    score: float
    steps_taken: int
    tokens: List[int]
    is_finished: bool = False
    uncertainty: float = 0.0
    sub_goals_completed: int = 0
    reflections: List[str] = field(default_factory=list)


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


# ==============================================================================
# MCTS Node
# ==============================================================================

@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    hidden: torch.Tensor
    score: float = 0.0
    visits: int = 0
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    depth: int = 0
    action: int = 0

    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for exploration."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.score / self.visits
        exploration = math.sqrt(2 * math.log(max(1, self.parent.visits if self.parent else 1)) / self.visits)
        return exploitation + exploration


# ==============================================================================
# Reflexion Module
# ==============================================================================

class ReflexionModule(nn.Module):
    """Learns from mistakes by reflecting on failed reasoning paths."""

    def __init__(self, dim: int = 8192):
        super().__init__()
        self.dim = dim

        # Mistake detector
        self.mistake_detector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        # Reflection generator
        self.reflection_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Lesson accumulator
        self.lesson_memory = nn.GRU(dim, dim, batch_first=True)

        # Lesson gate (how much to trust past reflections)
        self.lesson_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        current: torch.Tensor,
        initial: torch.Tensor,
        lesson_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reflect on current state and generate improvement.

        Args:
            current: Current hidden state [batch, dim]
            initial: Initial hidden state [batch, dim]
            lesson_history: [batch, n_lessons, dim] past reflections

        Returns:
            reflection: [batch, dim] improved representation
            mistake_score: [batch] probability of mistake
            updated_lessons: [batch, n_lessons, dim] updated lesson memory
        """
        # Detect mistakes
        mistake_input = torch.cat([current, initial], dim=-1)
        mistake_score = self.mistake_detector(mistake_input).squeeze(-1)

        # Generate reflection
        reflection = self.reflection_net(current)

        # Apply lessons from memory
        if lesson_history is not None and lesson_history.shape[1] > 0:
            # Attend over past lessons
            lesson_context = lesson_history.mean(dim=1)  # [batch, dim]
            gate = self.lesson_gate(torch.cat([reflection, lesson_context], dim=-1))
            reflection = gate * reflection + (1 - gate) * lesson_context

        # Update lesson memory
        if lesson_history is None:
            lesson_history = reflection.unsqueeze(1)
        else:
            lesson_history = torch.cat([lesson_history, reflection.unsqueeze(1)], dim=1)

        return reflection, mistake_score, lesson_history


# ==============================================================================
# Planner Module
# ==============================================================================

class PlannerModule(nn.Module):
    """Decomposes complex problems into sub-goals and tracks progress."""

    def __init__(self, dim: int = 8192, max_sub_goals: int = 8):
        super().__init__()
        self.dim = dim
        self.max_sub_goals = max_sub_goals

        # Sub-goal generator
        self.subgoal_generator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * max_sub_goals),
        )

        # Sub-goal selector (which sub-goal to focus on)
        self.subgoal_selector = nn.Sequential(
            nn.Linear(dim, max_sub_goals),
            nn.Softmax(dim=-1),
        )

        # Progress tracker per sub-goal
        self.progress_trackers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.SiLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid(),
            ) for _ in range(max_sub_goals)
        ])

        # Sub-goal completion detector
        self.completion_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, max_sub_goals),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        sub_goals: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Plan and track progress through sub-goals.

        Args:
            hidden: Current state [batch, dim]
            sub_goals: [batch, n_goals, dim] existing sub-goals

        Returns:
            Dictionary with sub_goals, active_goal, progress, completion
        """
        B = hidden.shape[0]

        # Generate sub-goals if not provided
        if sub_goals is None:
            sub_goals_raw = self.subgoal_generator(hidden)  # [batch, dim * max_sub_goals]
            sub_goals = sub_goals_raw.view(B, self.max_sub_goals, self.dim)

        # Select active sub-goal
        active_weights = self.subgoal_selector(hidden)  # [batch, max_sub_goals]
        active_idx = active_weights.argmax(dim=-1)  # [batch]

        # Track progress for each sub-goal
        progress_list = []
        for i, tracker in enumerate(self.progress_trackers):
            p = tracker(torch.cat([hidden, sub_goals[:, i]], dim=-1))
            progress_list.append(p)
        progress = torch.cat(progress_list, dim=-1)  # [batch, max_sub_goals]

        # Check completion
        completion = self.completion_detector(hidden)  # [batch, max_sub_goals]

        return {
            "sub_goals": sub_goals,
            "active_weights": active_weights,
            "active_idx": active_idx,
            "progress": progress,
            "completion": completion,
            "completed_count": (completion > 0.5).float().sum(dim=-1),
        }


# ==============================================================================
# Enhanced DeepSearch Module
# ==============================================================================

class DeepSearchModule(nn.Module):
    """
    DeepSearch v2 - Advanced reasoning engine

    Features:
      1. Beam search over reasoning chains
      2. MCTS-style exploration for complex problems
      3. Reflexion: learn from mistakes
      4. Planning: decompose into sub-goals
      5. Uncertainty-aware: explore when uncertain
    """

    def __init__(
        self,
        dim: int = 8192,
        n_heads: int = 16,
        max_depth: int = 10,
        beam_width: int = 4,
        n_thoughts_per_step: int = 4,
        completion_threshold: float = 0.8,
        use_mcts: bool = True,
        use_reflexion: bool = True,
        use_planning: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.n_thoughts = n_thoughts_per_step
        self.completion_threshold = completion_threshold
        self.use_mcts = use_mcts
        self.use_reflexion = use_reflexion
        self.use_planning = use_planning

        # Core components
        self.thought_generator = ThoughtGenerator(dim, n_heads, max_thoughts=8)
        self.evaluator = SearchEvaluator(dim)

        # State projection
        self.state_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # Step embedding
        self.step_embedding = nn.Embedding(max_depth, dim)

        # Output merger
        self.output_merger = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Enhanced components
        if use_reflexion:
            self.reflexion = ReflexionModule(dim)
        if use_planning:
            self.planner = PlannerModule(dim)

        # Uncertainty estimator
        self.uncertainty_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
            nn.Softplus(),
        )

        # MCTS expansion network
        if use_mcts:
            self.mcts_expansion = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )

    def _expand_mcts(self, node: MCTSNode, initial: torch.Tensor) -> List[MCTSNode]:
        """Expand MCTS node with new children."""
        step_emb = self.step_embedding(
            torch.tensor(min(node.depth, self.max_depth - 1), device=node.hidden.device)
        ).unsqueeze(0)

        state = node.hidden + step_emb
        thoughts, weights, _ = self.thought_generator(state, self.n_thoughts)

        children = []
        for i in range(self.n_thoughts):
            thought = thoughts[:, i]
            eval_result = self.evaluator(thought, initial)
            score = eval_result["value"] + 0.5 * eval_result["progress"]

            child = MCTSNode(
                hidden=self.mcts_expansion(thought),
                score=score.mean().item(),
                parent=node,
                depth=node.depth + 1,
                action=i,
            )
            children.append(child)

        return children

    def _mcts_search(self, initial_hidden: torch.Tensor, n_simulations: int = 16) -> Dict[str, torch.Tensor]:
        """Run MCTS-style search."""
        device = initial_hidden.device
        B = initial_hidden.shape[0]

        root = MCTSNode(hidden=initial_hidden, visits=1)
        best_node = root
        best_score = float('-inf')

        for _ in range(n_simulations):
            # Selection: traverse tree using UCB
            node = root
            while node.children:
                node = max(node.children, key=lambda n: n.ucb_score)

            # Expansion: add children
            if node.visits > 0 and node.depth < self.max_depth:
                children = self._expand_mcts(node, initial_hidden)
                node.children.extend(children)

            # Simulation: evaluate leaf
            if node.children:
                node = max(node.children, key=lambda n: n.ucb_score)

            eval_result = self.evaluator(node.hidden, initial_hidden)
            score = eval_result["value"].mean().item()

            # Backpropagation
            current = node
            while current is not None:
                current.visits += 1
                current.score += score
                current = current.parent

            if score > best_score:
                best_score = score
                best_node = node

        return {
            "best_reasoning": best_node.hidden,
            "best_hidden": best_node.hidden,
            "best_score": torch.tensor(best_score, device=device),
            "depth_reached": best_node.depth,
            "all_scores": [],
        }

    def beam_search(
        self,
        initial_hidden: torch.Tensor,
        n_thoughts: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Run beam search reasoning with reflexion and planning.

        Args:
            initial_hidden: Starting state [batch, dim]
            n_thoughts: Thoughts per step

        Returns:
            Dictionary with best_reasoning, all_scores, depth_reached
        """
        batch_size = initial_hidden.shape[0]
        device = initial_hidden.device

        # Initialize beam
        initial_eval = self.evaluator(initial_hidden, initial_hidden)
        beams = [{
            "hidden": initial_hidden,
            "score": initial_eval["value"],
            "depth": 0,
            "accumulated": initial_hidden,
            "lessons": None,
            "sub_goals": None,
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

                # Reflexion: reflect on current state
                if self.use_reflexion:
                    state, mistake_score, lessons = self.reflexion(
                        state, initial_hidden, beam.get("lessons")
                    )
                else:
                    lessons = beam.get("lessons")

                # Planning: decompose into sub-goals
                if self.use_planning:
                    plan_result = self.planner(state, beam.get("sub_goals"))
                    sub_goals = plan_result["sub_goals"]
                    # Focus on active sub-goal
                    active_idx = plan_result["active_idx"]
                    state = state + sub_goals[torch.arange(batch_size), active_idx] * 0.1
                else:
                    sub_goals = beam.get("sub_goals")

                # Generate thoughts
                thoughts, weights, div_loss = self.thought_generator(state, n_thoughts)

                # Evaluate each thought
                for i in range(n_thoughts):
                    thought = thoughts[:, i]
                    eval_result = self.evaluator(thought, initial_hidden)

                    # Uncertainty-aware scoring
                    uncertainty = self.uncertainty_net(thought).squeeze(-1)
                    exploration_bonus = 0.1 * uncertainty  # Explore more when uncertain

                    depth_penalty = 0.02 * depth
                    score = eval_result["value"] + 0.5 * eval_result["progress"] - depth_penalty + exploration_bonus

                    combined_score = beam["score"] + score * weights[:, i]

                    candidate = {
                        "hidden": self.state_proj(thought),
                        "score": combined_score,
                        "depth": depth + 1,
                        "accumulated": self.output_merger(
                            torch.cat([beam["accumulated"], thought], dim=-1)
                        ),
                        "lessons": lessons,
                        "sub_goals": sub_goals,
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

            # Keep top candidates
            candidates.sort(key=lambda c: c["score"].mean().item(), reverse=True)
            beams = candidates[:self.beam_width]

            all_scores.append({
                "depth": depth,
                "best_score": beams[0]["score"].mean().item(),
                "n_candidates": len(candidates),
            })

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
            use_search: Whether to use search or single pass
            n_thoughts: Thoughts per step

        Returns:
            Dictionary with output, scores, depth
        """
        if hidden.dim() == 3:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        if use_search and self.training is False:
            # Use MCTS for complex problems, beam search otherwise
            if self.use_mcts and pooled.shape[0] <= 4:
                result = self._mcts_search(pooled, n_simulations=16)
            else:
                result = self.beam_search(pooled, n_thoughts)
        else:
            # Single-step during training
            thoughts, weights, div_loss = self.thought_generator(pooled, n_thoughts)
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
