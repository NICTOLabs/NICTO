"""
Memory System
Hierarchical memory with working, episodic, semantic, and procedural memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from collections import deque
import numpy as np


class WorkingMemory(nn.Module):
    """
    Working Memory - Current conversation context
    
    Like human working memory, holds current information
    for active processing (capacity: ~7 items)
    """
    
    def __init__(self, dim: int = 4096, capacity: int = 1000):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        
        # Memory buffer
        self.register_buffer(
            "memory_buffer",
            torch.zeros(1, capacity, dim)
        )
        
        # Attention for retrieval
        self.retrieval_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True,
        )
        
        # Update gate
        self.update_gate = nn.Linear(dim * 2, dim)
        
        # Position encoding
        self.position_encoding = nn.Embedding(capacity, dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "read_write",
    ) -> Dict[str, torch.Tensor]:
        """
        Access working memory
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mode: "read", "write", or "read_write"
            
        Returns:
            Dictionary with memory outputs
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Add position encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_encoding(positions)
        x_pos = x + pos_emb
        
        outputs = {}
        
        if mode in ["read", "read_write"]:
            # Read from memory
            memory = self.memory_buffer.expand(batch_size, -1, -1)
            
            # Use attention for retrieval
            retrieved, attention_weights = self.retrieval_attention(
                x_pos, memory, memory
            )
            
            outputs["retrieved"] = retrieved
            outputs["attention_weights"] = attention_weights
        
        if mode in ["write", "read_write"]:
            # Write to memory
            x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, dim]
            mem_head = self.memory_buffer.expand(batch_size, -1, -1)[:, :1, :]  # [B, 1, dim]
            update_input = torch.cat([x_mean, mem_head], dim=-1)  # [B, 1, 2*dim]
            update_gate = torch.sigmoid(self.update_gate(update_input))  # [B, 1, dim]

            # Shift memory and add new (update first sample for simplicity)
            new_item = x_mean  # [B, 1, dim]
            self.memory_buffer.data = torch.cat(
                [new_item[:1], self.memory_buffer.data[:, :-1, :]], dim=1
            )
        
        return outputs


class EpisodicMemory(nn.Module):
    """
    Episodic Memory - Shared experiences

    Stores and retrieves specific experiences:
    - Conversations
    - Interactions
    - Emotional moments

    Uses a ring buffer to manage memory efficiently.
    """

    def __init__(self, dim: int = 4096, capacity: int = 8192):
        super().__init__()
        self.dim = dim
        self.capacity = capacity

        # Memory storage (ring buffer)
        self.memory_keys = nn.Parameter(torch.randn(capacity, dim) * 0.01)
        self.memory_values = nn.Parameter(torch.randn(capacity, dim) * 0.01)

        # Memory metadata (emotional valence, importance)
        self.memory_metadata = nn.Parameter(torch.zeros(capacity, 4))

        # Retrieval network
        self.retrieval_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Current memory pointer
        self.register_buffer("memory_pointer", torch.tensor(0))
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "retrieve",
        top_k: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Access episodic memory
        
        Args:
            x: Input tensor
            mode: "store", "retrieve", or "both"
            top_k: Number of memories to retrieve
            
        Returns:
            Dictionary with memory outputs
        """
        batch_size = x.shape[0]
        x_query = self.retrieval_net(x.mean(dim=1))
        
        outputs = {}
        
        if mode in ["retrieve", "both"]:
            # Only search through filled memories
            n_filled = min(int(self.memory_pointer.item()), self.capacity)
            if n_filled == 0:
                outputs["retrieved"] = x.mean(dim=1)
                outputs["similarities"] = torch.zeros(batch_size, 1, device=x.device)
                outputs["indices"] = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
                outputs["metadata"] = torch.zeros(batch_size, 1, 4, device=x.device)
            else:
                search_keys = self.memory_keys[:n_filled]  # [n_filled, dim]
                similarities = F.cosine_similarity(
                    x_query.unsqueeze(1),
                    search_keys.unsqueeze(0),
                    dim=-1
                )  # [B, n_filled]

                # Get top-k (limited by n_filled)
                k = min(top_k, n_filled)
                top_k_values, top_k_indices = torch.topk(similarities, k, dim=-1)

                # Get retrieved memories
                retrieved_values = self.memory_values[:n_filled][top_k_indices]
                retrieved_metadata = self.memory_metadata[:n_filled][top_k_indices]

                # Weight by similarity
                weights = F.softmax(top_k_values, dim=-1).unsqueeze(-1)
                retrieved = (retrieved_values * weights).sum(dim=1)

                outputs["retrieved"] = retrieved
                outputs["similarities"] = top_k_values
                outputs["indices"] = top_k_indices
                outputs["metadata"] = retrieved_metadata
        
        if mode in ["store", "both"]:
            # Store new memory
            importance = self.importance_scorer(x.mean(dim=1))
            
            # Only store if important enough
            if importance.mean() > 0.5:
                idx = self.memory_pointer % self.capacity
                self.memory_keys.data[idx] = x.mean(dim=1).mean(dim=0)
                self.memory_values.data[idx] = x.mean(dim=1).mean(dim=0)
                self.memory_metadata.data[idx] = torch.tensor([
                    importance.mean().detach().item(),
                    0.0,
                    0.0,
                    0.0,
                ])
                self.memory_pointer += 1
        
        return outputs


class SemanticMemory(nn.Module):
    """
    Semantic Memory - Facts and knowledge

    Stores general knowledge and facts:
    - World knowledge
    - User preferences
    - Learned patterns
    """

    def __init__(self, dim: int = 4096, n_facts: int = 8192):
        super().__init__()
        self.dim = dim
        self.capacity = n_facts

        # Fact storage (ring buffer)
        self.fact_keys = nn.Parameter(torch.randn(n_facts, dim) * 0.01)
        self.fact_values = nn.Parameter(torch.randn(n_facts, dim) * 0.01)
        self.fact_confidence = nn.Parameter(torch.ones(n_facts) * 0.5)

        self.register_buffer("fact_pointer", torch.tensor(0))
        
        # Retrieval network
        self.retrieval_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Fact updater
        self.fact_updater = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "query",
        top_k: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Access semantic memory
        
        Args:
            x: Input tensor
            mode: "query", "update", or "both"
            top_k: Number of facts to retrieve
            
        Returns:
            Dictionary with memory outputs
        """
        x_query = self.retrieval_net(x.mean(dim=1))
        
        outputs = {}
        
        if mode in ["query", "both"]:
            batch_size = x.shape[0]
            # Only search through filled facts
            n_filled = min(int(self.fact_pointer.item()), self.capacity)
            if n_filled == 0:
                outputs["retrieved"] = x.mean(dim=1)
                outputs["confidence"] = torch.ones(batch_size, 1, device=x.device) * 0.5
            else:
                search_keys = self.fact_keys[:n_filled]
                search_conf = self.fact_confidence[:n_filled]
                similarities = F.cosine_similarity(
                    x_query.unsqueeze(1),
                    search_keys.unsqueeze(0),
                    dim=-1
                )

                # Weight by confidence
                weighted_similarities = similarities * search_conf.unsqueeze(0)

                k = min(top_k, n_filled)
                top_k_values, top_k_indices = torch.topk(weighted_similarities, k, dim=-1)

                retrieved_facts = self.fact_values[:n_filled][top_k_indices]
                weights = F.softmax(top_k_values, dim=-1).unsqueeze(-1)
                retrieved = (retrieved_facts * weights).sum(dim=1)

                outputs["retrieved"] = retrieved
                outputs["confidence"] = search_conf[top_k_indices]
        
        return outputs


class HierarchicalMemory(nn.Module):
    """
    Complete Hierarchical Memory System
    
    Combines:
    - Working Memory (current context)
    - Episodic Memory (experiences)
    - Semantic Memory (facts)
    - Procedural Memory (skills)
    """
    
    def __init__(self, dim: int = 4096, episodic_capacity: int = 8192, semantic_capacity: int = 8192):
        super().__init__()
        self.dim = dim

        # Memory components
        self.working_memory = WorkingMemory(dim, capacity=min(1000, episodic_capacity))
        self.episodic_memory = EpisodicMemory(dim, capacity=episodic_capacity)
        self.semantic_memory = SemanticMemory(dim, n_facts=semantic_capacity)
        
        # Memory controller
        self.controller = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Memory fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        access_pattern: str = "full",
    ) -> Dict[str, torch.Tensor]:
        """
        Access hierarchical memory
        
        Args:
            x: Input tensor
            access_pattern: "working", "episodic", "semantic", or "full"
            
        Returns:
            Dictionary with memory outputs
        """
        outputs = {}
        
        if access_pattern in ["working", "full"]:
            working_out = self.working_memory(x, mode="read_write")
            outputs["working"] = working_out.get("retrieved", x)
        
        if access_pattern in ["episodic", "full"]:
            episodic_out = self.episodic_memory(x, mode="both")
            outputs["episodic"] = episodic_out.get("retrieved", x.mean(dim=1, keepdim=True).expand_as(x))
        
        if access_pattern in ["semantic", "full"]:
            semantic_out = self.semantic_memory(x, mode="query")
            outputs["semantic"] = semantic_out.get("retrieved", x.mean(dim=1, keepdim=True).expand_as(x))
        
        # Fuse all memories
        if access_pattern == "full":
            working = outputs.get("working", x)
            episodic = outputs.get("episodic", x)
            semantic = outputs.get("semantic", x)
            
            # Ensure same shape for fusion
            if working.dim() == 2:
                working = working.unsqueeze(1).expand_as(x)
            if episodic.dim() == 2:
                episodic = episodic.unsqueeze(1).expand_as(x)
            if semantic.dim() == 2:
                semantic = semantic.unsqueeze(1).expand_as(x)
            
            fused = self.fusion(torch.cat([working, episodic, semantic], dim=-1))
            outputs["fused"] = fused
        
        return outputs
