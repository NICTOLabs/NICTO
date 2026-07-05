"""
Mamba - State Space Model
Linear-time sequence modeling with selective state spaces
Based on Mamba-3 architecture for efficient long-context processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model
    
    Key innovation: Input-dependent parameters (B, C, delta)
    This gives Mamba attention-like content awareness with O(N) complexity
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A parameter (log space for stability)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with selective scan
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            cache: Cached state for inference
            
        Returns:
            output: Output tensor
            new_cache: Updated state
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # [B, S, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # Each [B, S, d_inner]
        
        # Convolution
        x = x.transpose(1, 2)  # [B, d_inner, S]
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # [B, S, d_inner]
        x = F.silu(x)
        
        # Compute SSM parameters
        x_dbl = self.x_proj(x)  # [B, S, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project dt to d_inner
        dt = self.dt_proj(dt)  # [B, S, d_inner]
        dt = F.softplus(dt)
        
        # Get A from log space
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Selective scan
        y, new_cache = self._selective_scan(x, dt, A, B, C, cache)
        
        # Skip connection
        y = y * self.D
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output, new_cache
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Selective scan algorithm
        
        This is the core of Mamba - O(N) complexity with input-dependent parameters
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        if cache is not None:
            h = cache
        else:
            h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            # Get current timestep values
            x_t = x[:, t, :]  # [B, d_inner]
            dt_t = dt[:, t, :]  # [B, d_inner]
            B_t = B[:, t, :]  # [B, d_state]
            C_t = C[:, t, :]  # [B, d_state]
            
            # Discretize
            dA = torch.exp(dt_t.unsqueeze(-1) * A)  # [B, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, d_inner, d_state]
            
            # State update
            h = h * dA + dB * x_t.unsqueeze(-1)
            
            # Output
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, S, d_inner]
        
        return y, h


class MambaBlock(nn.Module):
    """Mamba block with residual connection"""
    
    def __init__(self, d_model: int, mamba_config: dict):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mamba = SelectiveSSM(d_model, **mamba_config)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        residual = x
        x = self.norm(x)
        x, new_cache = self.mamba(x, cache)
        x = residual + self.dropout(x)
        return x, new_cache


class MambaStack(nn.Module):
    """
    Stack of Mamba blocks for sequence modeling
    
    This provides linear-time processing for long contexts
    """
    
    def __init__(
        self,
        n_layers: int = 48,
        d_model: int = 4096,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                mamba_config={
                    "d_state": d_state,
                    "d_conv": d_conv,
                    "expand": expand,
                }
            ) for _ in range(n_layers)
        ])
        
        self.norm_f = nn.RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all layers
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            caches: List of cached states for each layer
            
        Returns:
            output: Output tensor
            new_caches: Updated caches
        """
        new_caches = []
        
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x, new_cache = layer(x, cache)
            new_caches.append(new_cache)
        
        x = self.norm_f(x)
        
        return x, new_caches
