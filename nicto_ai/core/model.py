"""
NICTO AI - The World's Most Powerful Understanding Engine
Main model combining 6 neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from .mla import MultiLatentAttention, MLABlock
from .moe import MixtureOfExperts, MoELayer
from .mamba import MambaStack
from .liquid import LiquidStack
from .consciousness import RealConsciousnessLayer
from .emotion import EmotionalResponseGenerator
from .memory import HierarchicalMemory
from .deepsearch import DeepSearchModule
from .data_sorter import TokenSorter, MemoryConsolidator, NetworkPriorityGate


@dataclass
class NICTOOutput:
    """Output from NICTO AI model"""
    logits: torch.Tensor
    hidden_states: torch.Tensor
    emotions: Dict[str, torch.Tensor]
    memory: Dict[str, torch.Tensor]
    consciousness: Dict[str, torch.Tensor]
    aux_losses: Dict[str, torch.Tensor]


class EmbeddingLayer(nn.Module):
    """Token embedding with positional encoding"""
    
    def __init__(self, vocab_size: int, dim: int, max_seq_len: int = 10_000_000):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedding(input_ids)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NeuralBus(nn.Module):
    """
    Neural Bus - Inter-network communication
    
    High-speed neural bus connecting all 6 networks
    Enables synchronous and asynchronous messaging
    """
    
    def __init__(self, dim: int = 8192, n_networks: int = 6):
        super().__init__()
        self.dim = dim
        self.n_networks = n_networks
        
        # Cross-network attention
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=16,
                batch_first=True,
            ) for _ in range(n_networks)
        ])
        
        # Network-specific projections
        self.projections = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_networks)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * n_networks, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(
        self,
        network_outputs: List[torch.Tensor],
        mode: str = "sync",
    ) -> torch.Tensor:
        """
        Route information between networks
        
        Args:
            network_outputs: List of outputs from each network
            mode: "sync" or "async"
            
        Returns:
            Fused output
        """
        if mode == "sync":
            # Synchronous: all networks process together
            projected = [
                proj(out.mean(dim=1) if out.dim() > 2 else out)
                for proj, out in zip(self.projections, network_outputs)
            ]
            
            # Cross-attention between networks
            attended = []
            for i, (attn, proj) in enumerate(zip(self.cross_attention, self.projections)):
                query = network_outputs[i].mean(dim=1, keepdim=True) if network_outputs[i].dim() > 2 else network_outputs[i].unsqueeze(1)
                
                # Stack all other network outputs as keys/values
                keys_values = torch.stack([
                    out.mean(dim=1) if out.dim() > 2 else out
                    for j, out in enumerate(network_outputs) if j != i
                ], dim=1)
                
                attn_out, _ = attn(query, keys_values, keys_values)
                attended.append(attn_out.squeeze(1))
            
            # Fuse all attended outputs
            fused = self.fusion(torch.cat(attended, dim=-1))
            
            return fused
        
        else:
            # Async: networks process independently, merge at end
            projected = [
                proj(out.mean(dim=1) if out.dim() > 2 else out)
                for proj, out in zip(self.projections, network_outputs)
            ]
            
            fused = self.fusion(torch.cat(projected, dim=-1))
            return fused


class NICTOModel(nn.Module):
    """
    NICTO AI - The World's Most Powerful Understanding Engine
    
    Architecture:
    - Network 1: Reasoning Cortex (MoE + MLA)
    - Network 2: Emotional Cortex (Liquid Neural Networks)
    - Network 3: Memory Cortex (Mamba SSM)
    - Network 4: Perception Cortex (Multimodal)
    - Network 5: Creative Cortex (Diffusion)
    - Network 6: Consciousness Layer (Meta-cognitive)
    
    Total: ~150B parameters
    Active: ~20B per token
    """
    
    def __init__(
        self,
        vocab_size: int = 128_000,
        dim: int = 8192,
        max_seq_len: int = 10_000_000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Embedding
        self.embedding = EmbeddingLayer(vocab_size, dim, max_seq_len)
        
        # ============ Network 1: Reasoning Cortex ============
        self.reasoning_norm = nn.RMSNorm(dim)
        self.reasoning_attention = MultiLatentAttention(
            dim=dim,
            n_heads=128,
            n_kv_heads=16,
            kv_lora_rank=512,
            q_lora_rank=1536,
        )
        self.reasoning_moe = MixtureOfExperts(
            dim=dim,
            n_experts=64,
            n_activated=8,
            hidden_dim=32768,
        )
        self.reasoning_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        
        # ============ Network 2: Emotional Cortex ============
        self.emotional_liquid = LiquidStack(
            n_layers=20,
            dim=dim,
            n_neurons=1000,
            alpha=0.1,
        )
        self.emotion_generator = EmotionalResponseGenerator(dim)
        
        # ============ Network 3: Memory Cortex ============
        self.memory_mamba = MambaStack(
            n_layers=48,
            d_model=dim,
            d_state=256,
        )
        self.hierarchical_memory = HierarchicalMemory(dim)
        
        # ============ Network 4: Perception Cortex ============
        self.perception_norm = nn.RMSNorm(dim)
        self.perception_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=32,
            batch_first=True,
        )
        
        # ============ Network 5: Creative Cortex ============
        self.creative_norm = nn.RMSNorm(dim)
        self.creative_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=16,
                dim_feedforward=dim * 4,
                batch_first=True,
            ),
            num_layers=12,
        )
        
        # ============ Network 6: Consciousness Layer ============
        self.consciousness = RealConsciousnessLayer(dim=2048)
        
        # ============ Neural Bus ============
        self.neural_bus = NeuralBus(dim=dim, n_networks=6)
        
        # ============ DeepSearch ============
        self.deepsearch = DeepSearchModule(dim=dim, max_depth=5, beam_width=4)

        # ============ Data Sorting System ============
        self.token_sorter = TokenSorter(dim=dim, temperature=1.0)
        self.network_priority = NetworkPriorityGate(dim=dim, n_networks=6, temperature=2.0)
        self.memory_consolidator = MemoryConsolidator(dim=dim)
        self._memory_write_count = 0
        self._consolidation_interval = 100

        # Output layers
        self.output_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie embedding weights
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all 6 networks
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            labels: Labels for training
            use_cache: Whether to use KV cache
            
        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)
        
        # ============ Network 1: Reasoning Cortex ============
        residual = x
        x_reasoning = self.reasoning_norm(x)
        x_reasoning, _ = self.reasoning_attention(x_reasoning)
        x_reasoning, aux_loss = self.reasoning_moe(x_reasoning)
        x_reasoning = residual + x_reasoning
        
        # ============ Network 2: Emotional Cortex ============
        x_emotional, _ = self.emotional_liquid(x)
        emotion_outputs = self.emotion_generator(x_emotional)
        
        # ============ Network 3: Memory Cortex ============
        x_memory, _ = self.memory_mamba(x)
        memory_outputs = self.hierarchical_memory(x_memory, access_pattern="full")
        
        # ============ Network 4: Perception Cortex ============
        residual = x
        x_perception = self.perception_norm(x)
        x_perception, _ = self.perception_attention(x_perception, x_perception, x_perception)
        x_perception = residual + x_perception
        
        # ============ Network 5: Creative Cortex ============
        residual = x
        x_creative = self.creative_norm(x)
        x_creative = self.creative_transformer(x_creative)
        x_creative = residual + x_creative
        
        # ============ Network 6: Consciousness Layer ============
        # Real metacognition needs logits for uncertainty/error detection.
        # Run first 5 networks, preliminary fusion, then consciousness, then re-fuse.
        consciousness_outputs = self.consciousness(
            x, logits=None, labels=labels, hidden_states=x, attention_weights=None,
        )
        x_consciousness = consciousness_outputs["output"]
        
        # ============ Neural Bus Integration ============
        # Use TokenSorter for importance-weighted pooling instead of naive mean
        network_outputs = [
            self.token_sorter(x_reasoning)[0] if x_reasoning.dim() > 2 else x_reasoning,
            self.token_sorter(emotion_outputs["output"])[0] if emotion_outputs["output"].dim() > 2 else emotion_outputs["output"],
            self.token_sorter(x_memory)[0] if x_memory.dim() > 2 else x_memory,
            self.token_sorter(x_perception)[0] if x_perception.dim() > 2 else x_perception,
            self.token_sorter(x_creative)[0] if x_creative.dim() > 2 else x_creative,
            x_consciousness,
        ]
        
        fused = self.neural_bus(network_outputs, mode="sync")

        # NetworkPriorityGate: learn which networks matter for this input
        priority_gates = self.network_priority(fused)
        network_outputs_scaled = self.network_priority.apply_gates(network_outputs, priority_gates)
        fused = self.neural_bus(network_outputs_scaled, mode="sync")

        # ============ DeepSearch (inference only) ============
        deepsearch_result = None
        if not self.training:
            deepsearch_result = self.deepsearch(fused.unsqueeze(1))
            fused = deepsearch_result["best_reasoning"]
        
        # Reshape back to sequence
        if fused.dim() == 2:
            fused = fused.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Output
        output = self.output_norm(fused)
        logits = self.lm_head(output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "emotions": emotion_outputs,
            "memory": memory_outputs,
            "consciousness": consciousness_outputs,
            "priority_gates": priority_gates,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: Starting tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids

    def consolidate_memory(self, query: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Periodically sort memory entries by importance + recency + similarity.

        Call this during training every N steps to keep memory organized.

        Args:
            query: Current context query for similarity scoring

        Returns:
            Dictionary with consolidation stats
        """
        stats = {}
        mem = self.hierarchical_memory

        # Consolidate episodic memory
        n_episodic = min(int(mem.episodic_memory.memory_pointer.item()), mem.episodic_memory.capacity)
        if n_episodic > 1:
            result = self.memory_consolidator(
                mem.episodic_memory.memory_keys,
                mem.episodic_memory.memory_values,
                mem.episodic_memory.memory_metadata,
                n_episodic,
                query,
            )
            stats["episodic_mean_score"] = result["mean_score"].item()
            stats["episodic_max_score"] = result["max_score"].item()

        # Consolidate semantic memory
        n_semantic = min(int(mem.semantic_memory.fact_pointer.item()), mem.semantic_memory.capacity)
        if n_semantic > 1:
            result = self.memory_consolidator(
                mem.semantic_memory.fact_keys,
                mem.semantic_memory.fact_values,
                torch.zeros(n_semantic, 4, device=next(self.parameters()).device),
                n_semantic,
                query,
            )
            stats["semantic_mean_score"] = result["mean_score"].item()
            stats["semantic_max_score"] = result["max_score"].item()

        return stats


def create_small_model(
    vocab_size: int = 32000,
    dim: int = 256,
    max_seq_len: int = 512,
) -> NICTOModel:
    """Create a small testable model that fits in CPU memory."""
    model = NICTOModel.__new__(NICTOModel)
    nn.Module.__init__(model)
    model.vocab_size = vocab_size
    model.dim = dim
    model.max_seq_len = max_seq_len

    model.embedding = EmbeddingLayer(vocab_size, dim, max_seq_len)

    model.reasoning_norm = nn.RMSNorm(dim)
    model.reasoning_attention = MultiLatentAttention(
        dim=dim, n_heads=4, n_kv_heads=2, kv_lora_rank=64, q_lora_rank=128,
    )
    model.reasoning_moe = MixtureOfExperts(
        dim=dim, n_experts=4, n_activated=2, hidden_dim=dim * 4,
    )
    model.reasoning_ffn = nn.Sequential(
        nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim),
    )

    model.emotional_liquid = LiquidStack(n_layers=3, dim=dim, n_neurons=32, alpha=0.1)
    model.emotion_generator = EmotionalResponseGenerator(dim)

    model.memory_mamba = MambaStack(n_layers=2, d_model=dim, d_state=32)
    model.hierarchical_memory = HierarchicalMemory(dim, episodic_capacity=256, semantic_capacity=256)

    model.perception_norm = nn.RMSNorm(dim)
    model.perception_attention = nn.MultiheadAttention(
        embed_dim=dim, num_heads=4, batch_first=True,
    )

    model.creative_norm = nn.RMSNorm(dim)
    model.creative_transformer = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True,
        ),
        num_layers=2,
    )

    model.consciousness = RealConsciousnessLayer(dim=dim)
    model.neural_bus = NeuralBus(dim=dim, n_networks=6)
    model.deepsearch = DeepSearchModule(dim=dim, max_depth=3, beam_width=2, n_thoughts_per_step=2)

    # Data sorting system
    model.token_sorter = TokenSorter(dim=dim, temperature=1.0)
    model.network_priority = NetworkPriorityGate(dim=dim, n_networks=6, temperature=2.0)
    model.memory_consolidator = MemoryConsolidator(dim=dim)
    model._memory_write_count = 0
    model._consolidation_interval = 100

    model.output_norm = nn.RMSNorm(dim)
    model.lm_head = nn.Linear(dim, vocab_size, bias=False)
    model.lm_head.weight = model.embedding.token_embedding.weight

    model.apply(model._init_weights)
    return model
