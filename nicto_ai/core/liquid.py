"""
Liquid Neural Networks
Based on MIT CSAIL research - continuous-time adaptive networks
Real-time parameter adaptation for emotional processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LiquidNeuron(nn.Module):
    """
    Single liquid neuron with adaptive time constant
    
    Uses ODE-based dynamics that change based on input:
    dh/dt = -h/tau(x) + f(x, h)
    
    Where tau(x) is input-dependent time constant
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, alpha: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        
        # Input to hidden
        self.W_ih = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Hidden to hidden (recurrent)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Time constant predictor (input-dependent)
        self.tau_predictor = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Gate for stability
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
    
    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with continuous-time dynamics
        
        Args:
            x: Input [batch_size, input_dim]
            h: Hidden state [batch_size, hidden_dim]
            
        Returns:
            new_h: Updated hidden state
            delta_h: Change in hidden state (for monitoring)
        """
        batch_size = x.shape[0]
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # Compute time constant (adaptive)
        tau_input = torch.cat([x, h], dim=-1)
        tau = F.softplus(self.tau_predictor(tau_input)) + 1e-6  # Ensure positive
        
        # Compute gate
        gate = torch.sigmoid(self.gate(tau_input))
        
        # Input and recurrent contributions
        input_contribution = self.W_ih(x)
        recurrent_contribution = self.W_hh(h)
        
        # ODE update: dh/dt = -h/tau + f(x, h)
        dh_dt = -h / tau + F.silu(input_contribution + recurrent_contribution)
        
        # Euler step
        new_h = h + self.alpha * dh_dt
        new_h = gate * new_h  # Apply gate for stability
        
        delta_h = new_h - h
        
        return new_h, delta_h


class LiquidLayer(nn.Module):
    """
    Liquid layer with multiple neurons and adaptive dynamics
    
    This layer adapts its parameters in real-time based on input
    """
    
    def __init__(
        self,
        n_neurons: int = 1000,
        input_dim: int = 4096,
        output_dim: int = 4096,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, n_neurons, bias=False)
        
        # Liquid neurons
        self.neurons = LiquidNeuron(n_neurons, n_neurons, alpha)
        
        # Output projection
        self.output_proj = nn.Linear(n_neurons, output_dim, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through liquid layer
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            h: Hidden state [batch_size, n_neurons]
            
        Returns:
            output: Output tensor [batch_size, seq_len, output_dim]
            new_h: Updated hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, input_dim]
            
            # Project to neuron space
            x_t = self.input_proj(x_t)  # [B, n_neurons]
            
            # Liquid dynamics
            h, delta_h = self.neurons(x_t, h)
            
            # Project to output space
            out_t = self.output_proj(h)  # [B, output_dim]
            outputs.append(out_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [B, S, output_dim]
        output = self.norm(output)
        
        return output, h


class LiquidStack(nn.Module):
    """
    Stack of liquid layers for deep liquid neural network
    
    This provides:
    - Real-time adaptation
    - Continuous-time dynamics
    - Interpretable behavior
    """
    
    def __init__(
        self,
        n_layers: int = 20,
        dim: int = 4096,
        n_neurons: int = 1000,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        
        self.layers = nn.ModuleList([
            LiquidLayer(
                n_neurons=n_neurons,
                input_dim=dim,
                output_dim=dim,
                alpha=alpha,
            ) for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all liquid layers
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            states: List of hidden states for each layer
            
        Returns:
            output: Output tensor
            new_states: Updated hidden states
        """
        new_states = []
        
        for i, layer in enumerate(self.layers):
            h = states[i] if states is not None else None
            x, new_h = layer(x, h)
            new_states.append(new_h)
        
        x = self.norm_f(x)
        
        return x, new_states


class AdaptiveLiquidLayer(nn.Module):
    """
    Advanced liquid layer with multi-scale adaptation
    
    Features:
    - Multiple time constants (fast, medium, slow adaptation)
    - Gated information flow
    - Meta-learning capability
    """
    
    def __init__(
        self,
        dim: int = 4096,
        n_neurons: int = 1000,
        n_timescales: int = 3,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_timescales = n_timescales
        
        # Multi-scale liquid neurons
        self.fast_neuron = LiquidNeuron(dim, n_neurons, alpha * 2)  # Fast adaptation
        self.medium_neuron = LiquidNeuron(dim, n_neurons, alpha)     # Medium adaptation
        self.slow_neuron = LiquidNeuron(dim, n_neurons, alpha * 0.5) # Slow adaptation
        
        # Fusion gate
        self.fusion_gate = nn.Linear(dim + n_neurons * 3, dim)
        
        # Output projection
        self.output_proj = nn.Linear(n_neurons * 3, dim)
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with multi-scale adaptation
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            states: [fast_h, medium_h, slow_h]
            
        Returns:
            output: Output tensor
            new_states: [new_fast_h, new_medium_h, new_slow_h]
        """
        batch_size, seq_len, _ = x.shape
        
        if states is None:
            fast_h = medium_h = slow_h = None
        else:
            fast_h, medium_h, slow_h = states
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Multi-scale processing
            fast_h, _ = self.fast_neuron(x_t, fast_h)
            medium_h, _ = self.medium_neuron(x_t, medium_h)
            slow_h, _ = self.slow_neuron(x_t, slow_h)
            
            # Fuse multi-scale outputs
            fused = torch.cat([fast_h, medium_h, slow_h], dim=-1)
            gate_input = torch.cat([x_t, fused], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(gate_input))
            
            out_t = gate * self.output_proj(fused) + (1 - gate) * x_t
            outputs.append(out_t)
        
        output = torch.stack(outputs, dim=1)
        new_states = [fast_h, medium_h, slow_h]
        
        return output, new_states
