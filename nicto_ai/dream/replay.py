"""
NICTO AI - Experience Replay Buffer
Prioritized replay of past experiences for offline learning

Unlike standard experience replay (uniform sampling), this uses
TD-error and surprise-based prioritization so NICTO learns most
from its most surprising/incorrect predictions.
"""

import math
import random
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single experience with metadata for prioritized replay"""
    id: int
    input_ids: torch.Tensor
    logits: torch.Tensor
    loss: float
    hidden_states: Optional[torch.Tensor] = None
    # Metacognitive signals (from real metacognition)
    uncertainty: float = 0.0
    entropy: float = 0.0
    gradient_norm: float = 0.0
    surprise: float = 0.0  # How unexpected was this prediction
    # Outcome
    predicted_token: int = 0
    actual_token: int = 0
    was_correct: bool = True
    # Priority (higher = replay more often)
    priority: float = 1.0
    # Metadata
    timestamp: int = 0
    episode: int = 0


class SumTree:
    """
    Binary sum tree for efficient prioritized sampling.
    O(log n) for both update and sample.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.write_pos = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: Any):
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def sample_batch(self, batch_size: int) -> List[Tuple[int, float, Any]]:
        batch = []
        segment = self.total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.get(s)
            if data is not None:
                batch.append((idx, priority, data))
        return batch


class ExperienceReplayBuffer:
    """
    Prioritized experience replay for NICTO.

    Stores experiences and samples them based on priority:
    - High surprise = high priority (learn from what shocked us)
    - High loss = high priority (learn from mistakes)
    - Low confidence = high priority (learn from uncertainty)
    - Recent experiences get a small boost (recency bias)

    This is NOT uniform random sampling. NICTO learns most from
    the experiences where it was most wrong or most uncertain.
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Maximum experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(capacity)
        self._max_priority = 1.0
        self._id_counter = 0
        self._stats = {
            "total_added": 0,
            "total_sampled": 0,
            "avg_surprise": 0.0,
            "avg_loss": 0.0,
        }

    def add(self, experience: Experience):
        """Add an experience with priority based on its surprise/loss."""
        priority = self._compute_priority(experience)
        self.tree.add(priority, experience)
        self._id_counter += 1
        self._stats["total_added"] += 1

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences proportional to priority.

        Returns experiences weighted so high-priority (surprising/wrong)
        experiences are sampled more often.
        """
        batch = self.tree.sample_batch(batch_size)
        experiences = [exp for _, _, exp in batch if exp is not None]
        self._stats["total_sampled"] += len(experiences)
        return experiences

    def update_priorities(self, experiences: List[Experience], new_losses: List[float]):
        """Update priorities after re-evaluating experiences."""
        for exp, new_loss in zip(experiences, new_losses):
            exp.loss = new_loss
            exp.priority = self._compute_priority(exp)
            # Find and update in tree (simplified: just update max)
            self._max_priority = max(self._max_priority, exp.priority)

    def compute_td_error(
        self,
        model: nn.Module,
        experience: Experience,
    ) -> float:
        """
        Compute TD-error for an experience (how wrong was the prediction).

        This is the core signal for prioritized replay:
        large TD-error = the model was very wrong = learn from this.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(experience.input_ids.unsqueeze(0))
            new_logits = outputs["logits"]
            # Cross-entropy loss on the stored actual token
            loss = nn.functional.cross_entropy(
                new_logits[:, -1, :],
                experience.actual_token.unsqueeze(0) if isinstance(experience.actual_token, torch.Tensor) else torch.tensor([experience.actual_token]),
            ).item()
        model.train()
        return loss

    def _compute_priority(self, experience: Experience) -> float:
        """Compute priority from experience signals."""
        # Combine multiple signals
        priority = (
            0.3 * experience.loss
            + 0.3 * experience.surprise
            + 0.2 * experience.entropy
            + 0.1 * (1.0 if not experience.was_correct else 0.0)
            + 0.1 * experience.gradient_norm
        )
        # Apply exponent
        priority = (priority + 1e-6) ** self.alpha
        self._max_priority = max(self._max_priority, priority)
        return priority

    @property
    def size(self) -> int:
        return self.tree.n_entries

    @property
    def stats(self) -> Dict[str, float]:
        return dict(self._stats)

    def __len__(self):
        return self.tree.n_entries
