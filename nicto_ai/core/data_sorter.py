"""
NICTO AI - Data Sorting System
Intelligent data organization across all 6 neural networks and memory systems

Components:
- TokenSorter: Reorders tokens by learned importance before pooling
- MemoryConsolidator: Re-sorts memory entries by importance + recency + similarity
- NetworkPriorityGate: Learns which networks matter most per input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TokenSorter(nn.Module):
    """
    Reorders tokens by learned importance before pooling.

    Instead of naive mean-pooling (which loses sequence structure),
    this scores each token position and produces an importance-weighted
    pooling. The most informative positions contribute more to the output.
    """

    def __init__(self, dim: int = 8192, top_k_ratio: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.top_k_ratio = top_k_ratio
        self.temperature = temperature

        # Score each token position for importance
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )

        # Optional: project before scoring for better feature extraction
        self.pre_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sort and pool tokens by importance.

        Args:
            x: Token sequence [batch, seq_len, dim]
            return_scores: Whether to return importance scores

        Returns:
            pooled: [batch, dim] — importance-weighted pooling
            scores: [batch, seq_len] — importance scores (if return_scores)
        """
        if x.dim() == 2:
            # Already pooled, nothing to sort
            return x, None

        projected = self.pre_proj(x)
        scores = self.importance_scorer(projected).squeeze(-1)  # [B, seq_len]

        # Soft importance weights via temperature-scaled softmax
        soft_weights = F.softmax(scores * self.temperature, dim=-1)  # [B, seq_len]

        # Importance-weighted pooling
        pooled = (x * soft_weights.unsqueeze(-1)).sum(dim=1)  # [B, dim]

        if return_scores:
            return pooled, scores
        return pooled, None


class MemoryConsolidator(nn.Module):
    """
    Re-sorts memory entries by combining importance + recency + similarity.

    Runs periodically (not every forward pass) to reorganize EpisodicMemory
    and SemanticMemory. Entries that are more important, more recent, or
    more relevant to the current context are moved to lower indices (higher
    priority positions) in the ring buffer.
    """

    def __init__(self, dim: int = 8192):
        super().__init__()
        self.dim = dim

        # Learnable importance scorer for stored values
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Recency decay rate (learnable)
        self.decay_logit = nn.Parameter(torch.tensor(0.01))

    @property
    def decay_rate(self) -> torch.Tensor:
        """Decay rate between 0 and 1 (sigmoid ensures this)."""
        return torch.sigmoid(self.decay_logit)

    def compute_sort_scores(
        self,
        values: torch.Tensor,
        n_filled: int,
        query: Optional[torch.Tensor] = None,
        importance_weight: float = 0.4,
        recency_weight: float = 0.3,
        similarity_weight: float = 0.3,
    ) -> torch.Tensor:
        """
        Compute combined sort scores for memory entries.

        Args:
            values: [n_filled, dim] — stored memory values
            n_filled: Number of valid entries
            query: [batch, dim] — current context query (optional)
            importance_weight: Weight for importance score
            recency_weight: Weight for recency score
            similarity_weight: Weight for similarity score

        Returns:
            scores: [n_filled] — combined sort scores (higher = higher priority)
        """
        device = values.device

        # 1. Importance: learned score of each entry's value
        importance = self.importance_scorer(values[:n_filled]).squeeze(-1)  # [n_filled]

        # 2. Recency: newer entries score higher (exponential decay from most recent)
        positions = torch.arange(n_filled, device=device, dtype=torch.float)
        decay = self.decay_rate
        recency = decay ** (n_filled - 1 - positions)  # [n_filled] — most recent = highest

        # 3. Similarity to current query (if provided)
        if query is not None:
            # query: [1, dim] or [dim], values: [n_filled, dim]
            if query.dim() == 1:
                query = query.unsqueeze(0)
            # Normalize for cosine similarity
            query_norm = F.normalize(query, dim=-1)  # [1, dim]
            values_norm = F.normalize(values[:n_filled], dim=-1)  # [n_filled, dim]
            similarity = (values_norm @ query_norm.T).squeeze(-1)  # [n_filled]
            similarity = (similarity + 1) / 2  # shift to [0, 1]
        else:
            similarity = torch.ones(n_filled, device=device) * 0.5

        # Combined score
        scores = (
            importance_weight * importance
            + recency_weight * recency
            + similarity_weight * similarity
        )

        return scores

    def sort_memory(
        self,
        keys: nn.Parameter,
        values: nn.Parameter,
        metadata: nn.Parameter,
        n_filled: int,
        query: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Sort memory entries in-place by combined score.

        Args:
            keys: [capacity, dim] — memory keys parameter
            values: [capacity, dim] — memory values parameter
            metadata: [capacity, 4] — memory metadata parameter
            n_filled: Number of valid entries
            query: Current context query for similarity scoring
        """
        if n_filled <= 1:
            return

        scores = self.compute_sort_scores(values, n_filled, query)  # [n_filled]
        sorted_indices = torch.argsort(scores, descending=True)  # high score = low index

        # Reorder the filled portion
        with torch.no_grad():
            keys.data[:n_filled] = keys.data[:n_filled][sorted_indices].clone()
            values.data[:n_filled] = values.data[:n_filled][sorted_indices].clone()
            metadata.data[:n_filled] = metadata.data[:n_filled][sorted_indices].clone()

    def forward(
        self,
        keys: nn.Parameter,
        values: nn.Parameter,
        metadata: nn.Parameter,
        n_filled: int,
        query: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Consolidate and sort memory entries.

        Args:
            keys, values, metadata: Memory parameters
            n_filled: Number of valid entries
            query: Current context query

        Returns:
            Dictionary with sort_scores for monitoring
        """
        if n_filled <= 1:
            return {"sort_scores": torch.tensor(0.0)}

        scores = self.compute_sort_scores(values, n_filled, query)
        self.sort_memory(keys, values, metadata, n_filled, query)

        return {
            "sort_scores": scores,
            "mean_score": scores.mean(),
            "max_score": scores.max(),
            "min_score": scores.min(),
        }


class NetworkPriorityGate(nn.Module):
    """
    Learns which of the 6 networks matter most for the current input.

    Produces per-network importance weights that scale each network's
    contribution in the NeuralBus fusion. This allows the model to
    focus on the most relevant networks for each input, rather than
    treating all 6 equally.
    """

    def __init__(self, dim: int = 8192, n_networks: int = 6, temperature: float = 2.0):
        super().__init__()
        self.dim = dim
        self.n_networks = n_networks
        self.temperature = temperature

        # Score each network's relevance
        self.gate_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, n_networks),
        )

        # Learned baseline (what's "normal" importance for each network)
        self.register_buffer(
            "baseline_gates",
            torch.ones(n_networks) / n_networks,
        )

    def forward(
        self,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute network priority gates.

        Args:
            fused: NeuralBus fused output [batch, dim]

        Returns:
            gates: [batch, n_networks] — importance weights per network (sum to 1)
        """
        gate_logits = self.gate_network(fused)  # [B, n_networks]

        # Add baseline as a soft prior (prevents any gate from going to 0)
        gate_logits = gate_logits + torch.log(self.baseline_gates + 1e-8)

        gates = F.softmax(gate_logits * self.temperature, dim=-1)  # [B, n_networks]

        return gates

    def apply_gates(
        self,
        network_outputs: list,
        gates: torch.Tensor,
    ) -> list:
        """
        Scale network outputs by their priority gates.

        Args:
            network_outputs: List of [batch, dim] tensors (one per network)
            gates: [batch, n_networks] — priority weights

        Returns:
            List of scaled [batch, dim] tensors
        """
        scaled = []
        for i, out in enumerate(network_outputs):
            gate = gates[:, i : i + 1]  # [B, 1]
            scaled.append(out * gate)
        return scaled
