"""
Test NICTO AI Model
Verify all components work correctly
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nicto_ai.core.model import NICTOModel
from nicto_ai.configs.model_config import NICTOConfig


def test_model_initialization():
    """Test model can be initialized"""
    print("Testing NICTO AI Model Initialization...")
    
    # Small config for testing
    config = NICTOConfig(
        vocab_size=1000,
        dim=512,
    )
    
    model = NICTOModel(
        vocab_size=config.vocab_size,
        dim=config.dim,
        max_seq_len=1000,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def test_forward_pass(model):
    """Test forward pass"""
    print("\nTesting Forward Pass...")
    
    batch_size = 2
    seq_len = 32
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Emotions detected: {list(outputs['emotions'].keys())}")
    print(f"  Memory outputs: {list(outputs['memory'].keys())}")
    print(f"  Consciousness outputs: {list(outputs['consciousness'].keys())}")
    
    return outputs


def test_generation(model):
    """Test text generation"""
    print("\nTesting Text Generation...")
    
    # Create starting tokens
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
    
    print(f"  Input length: {input_ids.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  New tokens: {generated.shape[1] - input_ids.shape[1]}")
    
    return generated


def test_individual_components():
    """Test individual components"""
    print("\nTesting Individual Components...")
    
    dim = 256
    batch_size = 2
    seq_len = 16
    
    # Test MLA
    from nicto_ai.core.mla import MultiLatentAttention
    mla = MultiLatentAttention(dim=dim, n_heads=8, n_kv_heads=2)
    x = torch.randn(batch_size, seq_len, dim)
    mla_out, _ = mla(x)
    print(f"  MLA output shape: {mla_out.shape}")
    
    # Test MoE
    from nicto_ai.core.moe import MixtureOfExperts
    moe = MixtureOfExperts(dim=dim, n_experts=8, n_activated=2, hidden_dim=dim*4)
    moe_out, aux_loss = moe(x)
    print(f"  MoE output shape: {moe_out.shape}")
    
    # Test Mamba
    from nicto_ai.core.mamba import MambaStack
    mamba = MambaStack(n_layers=4, d_model=dim)
    mamba_out, _ = mamba(x)
    print(f"  Mamba output shape: {mamba_out.shape}")
    
    # Test Liquid
    from nicto_ai.core.liquid import LiquidStack
    liquid = LiquidStack(n_layers=4, dim=dim, n_neurons=128)
    liquid_out, _ = liquid(x)
    print(f"  Liquid output shape: {liquid_out.shape}")
    
    # Test Consciousness
    from nicto_ai.core.consciousness import RealConsciousnessLayer
    consciousness = RealConsciousnessLayer(dim=dim)
    consciousness_out = consciousness(x)
    print(f"  Consciousness output shape: {consciousness_out['output'].shape}")
    
    # Test Memory
    from nicto_ai.core.memory import HierarchicalMemory
    memory = HierarchicalMemory(dim=dim)
    memory_out = memory(x, access_pattern="full")
    print(f"  Memory fused shape: {memory_out['fused'].shape}")
    
    print("\n  All components working!")


def test_emotional_processing():
    """Test emotional processing"""
    print("\nTesting Emotional Processing...")
    
    from nicto_ai.core.emotion import EmotionalResponseGenerator
    
    dim = 256
    emotion_gen = EmotionalResponseGenerator(dim)
    
    x = torch.randn(2, 16, dim)
    emotion_outputs = emotion_gen(x)
    
    print(f"  Emotion shape: {emotion_outputs['emotion'].shape}")
    print(f"  Empathy types: {list(emotion_outputs['empathy'].keys())}")
    print(f"  Output shape: {emotion_outputs['output'].shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("NICTO AI - Model Test Suite")
    print("=" * 60)
    
    # Run tests
    model = test_model_initialization()
    outputs = test_forward_pass(model)
    generated = test_generation(model)
    test_individual_components()
    test_emotional_processing()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
