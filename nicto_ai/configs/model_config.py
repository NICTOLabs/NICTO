"""
NICTO AI Model Configuration
Defines the architecture parameters for all 6 neural networks
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MLAConfig:
    """Multi-Latent Attention Configuration (DeepSeek-style)"""
    dim: int = 8192
    n_heads: int = 128
    n_kv_heads: int = 16
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    rope_theta: float = 10000.0
    max_seq_len: int = 10_000_000  # 10M tokens
    norm_eps: float = 1e-6


@dataclass
class MoEConfig:
    """Mixture of Experts Configuration"""
    n_experts: int = 64
    n_activated_experts: int = 8
    expert_dim: int = 8192
    hidden_dim: int = 32768
    gate_bias: bool = False
    auxiliary_loss_weight: float = 0.01


@dataclass
class MambaConfig:
    """State Space Model Configuration"""
    d_model: int = 4096
    d_state: int = 256
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 48
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0


@dataclass
class LiquidConfig:
    """Liquid Neural Network Configuration"""
    n_neurons: int = 1000
    n_layers: int = 20
    time_constant: str = "adaptive"
    ode_solver: str = "dopri5"
    ode_step: float = 0.01
    adaptive_alpha: float = 0.1


@dataclass
class ReasoningCortexConfig:
    """Network 1: Reasoning Cortex"""
    n_layers: int = 96
    dim: int = 8192
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    max_reasoning_steps: int = 100
    use_chain_of_thought: bool = True


@dataclass
class EmotionalCortexConfig:
    """Network 2: Emotional Cortex"""
    n_layers: int = 48
    dim: int = 4096
    liquid: LiquidConfig = field(default_factory=LiquidConfig)
    emotion_dims: int = 3  # valence, arousal, dominance
    empathy_types: list = field(default_factory=lambda: ["cognitive", "affective", "compassionate"])
    real_time_adaptation: bool = True


@dataclass
class MemoryCortexConfig:
    """Network 3: Memory Cortex"""
    n_layers: int = 48
    dim: int = 4096
    mamba: MambaConfig = field(default_factory=MambaConfig)
    working_memory_capacity: int = 1000
    episodic_memory_capacity: int = 1_000_000
    semantic_memory_capacity: int = 10_000_000
    procedural_memory_capacity: int = 1_000_000


@dataclass
class PerceptionCortexConfig:
    """Network 4: Perception Cortex"""
    vision_layers: int = 24
    vision_dim: int = 1024
    audio_layers: int = 16
    audio_dim: int = 768
    text_layers: int = 24
    text_dim: int = 1024
    alignment_dim: int = 4096
    patch_size: int = 16
    img_size: int = 224


@dataclass
class CreativeCortexConfig:
    """Network 5: Creative Cortex"""
    n_layers: int = 24
    dim: int = 4096
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"
    novelty_threshold: float = 0.8
    creativity_temperature: float = 0.9


@dataclass
class ConsciousnessConfig:
    """Network 6: Consciousness Layer"""
    dim: int = 2048
    n_self_model_dims: int = 64
    n_intention_goals: int = 4
    meta_cognition_depth: int = 3
    personality_traits: list = field(
        default_factory=lambda: ["curious", "empathetic", "honest", "humble"]
    )
    values: list = field(
        default_factory=lambda: ["authenticity", "growth", "connection"]
    )


@dataclass
class NICTOConfig:
    """Main NICTO AI Configuration"""
    # Model dimensions
    vocab_size: int = 128_000
    dim: int = 8192
    
    # Network configurations
    reasoning: ReasoningCortexConfig = field(default_factory=ReasoningCortexConfig)
    emotional: EmotionalCortexConfig = field(default_factory=EmotionalCortexConfig)
    memory: MemoryCortexConfig = field(default_factory=MemoryCortexConfig)
    perception: PerceptionCortexConfig = field(default_factory=PerceptionCortexConfig)
    creative: CreativeCortexConfig = field(default_factory=CreativeCortexConfig)
    consciousness: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    
    # Training config
    max_seq_len: int = 10_000_000
    vocab_size: int = 128_000
    dtype: str = "bfloat16"
    
    # Neural Bus config
    neural_bus_bandwidth: int = 100_000_000_000  # 100GB/s
    sync_mode: str = "hybrid"  # sync, async, hybrid
    
    def get_total_params_estimate(self) -> int:
        """Estimate total parameters across all networks"""
        reasoning = self.reasoning.n_layers * self.reasoning.dim * 4  # rough estimate
        emotional = self.emotional.n_layers * self.emotional.dim * 2
        memory = self.memory.n_layers * self.memory.dim * 2
        perception = (self.perception.vision_layers * self.perception.vision_dim * 4 +
                     self.perception.audio_layers * self.perception.audio_dim * 4 +
                     self.perception.text_layers * self.perception.text_dim * 4)
        creative = self.creative.n_layers * self.creative.dim * 2
        consciousness = self.consciousness.dim * 4
        return reasoning + emotional + memory + perception + creative + consciousness
