"""
Mixture of Experts (MoE)
Based on DeepSeek-V3 with auxiliary-loss-free load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    """Single expert network - SwiGLU FFN"""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKRouter(nn.Module):
    """
    Top-K routing with auxiliary-loss-free load balancing
    
    DeepSeek-V3 innovation: No auxiliary loss needed for balanced routing
    Uses bias term for load balancing instead of penalty loss
    """
    
    def __init__(self, dim: int, n_experts: int, n_activated: int, bias: bool = False):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated = n_activated
        
        self.gate = nn.Linear(dim, n_experts, bias=bias)
        self.noise_gate = nn.Linear(dim, n_experts, bias=False)
        
        # Learnable bias for load balancing (auxiliary-loss-free)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts)) if bias else None
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and indices
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            weights: Routing weights [batch_size, seq_len, n_activated]
            indices: Expert indices [batch_size, seq_len, n_activated]
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute gate scores
        gate_logits = self.gate(x)  # [B, S, n_experts]
        
        # Add noise during training for exploration
        if self.training:
            noise = F.softplus(self.noise_gate(x))
            gate_logits = gate_logits + noise * torch.randn_like(gate_logits)
        
        # Apply bias for load balancing
        if self.expert_bias is not None:
            gate_logits = gate_logits + self.expert_bias
        
        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.n_activated, dim=-1)
        
        # Compute weights
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute auxiliary loss (optional, for monitoring)
        aux_loss = self._compute_load_balance_loss(gate_logits, top_k_indices)
        
        return top_k_weights, top_k_indices, aux_loss
    
    def _compute_load_balance_loss(
        self, gate_logits: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing loss for monitoring"""
        # Count expert usage
        expert_counts = torch.zeros(self.n_experts, device=gate_logits.device)
        for i in range(self.n_experts):
            expert_counts[i] = (indices == i).float().sum()
        
        # Compute load balance loss
        n_tokens = gate_logits.shape[0] * gate_logits.shape[1]
        expert_fraction = expert_counts / n_tokens
        gate_fraction = F.softmax(gate_logits, dim=-1).mean(dim=[0, 1])
        
        loss = (expert_fraction * gate_fraction).sum() * self.n_experts
        return loss


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with efficient routing
    
    Architecture:
    - N expert networks (e.g., 64 experts)
    - Router selects top-K experts per token (e.g., 8 of 64)
    - Only K experts are activated per token
    - SwiGLU activation for each expert
    """
    
    def __init__(
        self,
        dim: int = 8192,
        n_experts: int = 64,
        n_activated: int = 8,
        hidden_dim: int = 32768,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_activated = n_activated
        
        # Router
        self.router = TopKRouter(dim, n_experts, n_activated, bias)
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(n_experts)
        ])
        
        # Shared expert (always activated)
        self.shared_expert = Expert(dim, hidden_dim)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse expert activation
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            output: MoE output
            aux_loss: Load balancing loss
        """
        batch_size, seq_len, dim = x.shape
        
        # Get routing weights and indices
        weights, indices, aux_loss = self.router(x)
        
        # Reshape for processing
        x_flat = x.view(-1, dim)  # [B*S, dim]
        weights_flat = weights.view(-1, self.n_activated)  # [B*S, K]
        indices_flat = indices.view(-1, self.n_activated)  # [B*S, K]
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each activated expert
        for k in range(self.n_activated):
            expert_idx = indices_flat[:, k]  # [B*S]
            expert_weight = weights_flat[:, k]  # [B*S]
            
            # Group tokens by expert
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask].unsqueeze(-1) * expert_output
        
        # Add shared expert contribution
        shared_output = self.shared_expert(x_flat)
        output = output + shared_output
        
        # Reshape back
        output = output.view(batch_size, seq_len, dim)
        
        return output, aux_loss


class MoELayer(nn.Module):
    """MoE layer with pre-normalization"""
    
    def __init__(self, dim: int, moe_config: dict):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.moe = MixtureOfExperts(dim=dim, **moe_config)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm(x)
        x, aux_loss = self.moe(x)
        x = residual + self.dropout(x)
        return x, aux_loss
