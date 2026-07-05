"""
NICTO AI Complete Test Suite
Tests all components including VGG16 and training pipeline
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_vgg16():
    """Test VGG16 vision encoder"""
    print("=" * 60)
    print("Testing VGG16 Vision Encoder")
    print("=" * 60)

    from nicto_ai.core.vision import VGG16Encoder, VGG16ForNICTO

    # Test basic VGG16
    print("\n1. Testing VGG16Encoder...")
    vgg16 = VGG16Encoder(pretrained=False)
    dummy_image = torch.randn(2, 3, 224, 224)
    features = vgg16(dummy_image)
    print(f"   Input: {dummy_image.shape}")
    print(f"   Output: {features.shape}")
    assert features.shape == (2, 25088), f"Expected (2, 25088), got {features.shape}"
    print("   [PASS]")

    # Test spatial features
    print("\n2. Testing spatial features...")
    spatial = vgg16.forward_spatial(dummy_image)
    print(f"   Spatial output: {spatial.shape}")
    assert spatial.shape == (2, 512, 7, 7), f"Expected (2, 512, 7, 7), got {spatial.shape}"
    print("   [PASS]")

    # Test VGG16ForNICTO
    print("\n3. Testing VGG16ForNICTO...")
    vgg_nicto = VGG16ForNICTO(nicto_dim=2048, pretrained=False)
    outputs = vgg_nicto(dummy_image)
    print(f"   Global features: {outputs['global_features'].shape}")
    print(f"   Patch features: {outputs['patch_features'].shape}")
    assert outputs['global_features'].shape == (2, 2048)
    assert outputs['patch_features'].shape == (2, 49, 2048)
    print("   [PASS]")

    print("\nVGG16 tests passed!")
    return True


def test_mla():
    """Test Multi-Latent Attention"""
    print("\n" + "=" * 60)
    print("Testing Multi-Latent Attention (MLA)")
    print("=" * 60)

    from nicto_ai.core.mla import MultiLatentAttention

    dim = 512
    mla = MultiLatentAttention(dim=dim, n_heads=8, n_kv_heads=2, kv_lora_rank=64, q_lora_rank=128)
    x = torch.randn(2, 16, dim)
    output, cache = mla(x)

    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    assert output.shape == x.shape
    print("   [PASS]")
    return True


def test_moe():
    """Test Mixture of Experts"""
    print("\n" + "=" * 60)
    print("Testing Mixture of Experts (MoE)")
    print("=" * 60)

    from nicto_ai.core.moe import MixtureOfExperts

    dim = 512
    moe = MixtureOfExperts(dim=dim, n_experts=8, n_activated=2, hidden_dim=dim * 4)
    x = torch.randn(2, 16, dim)
    output, aux_loss = moe(x)

    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Aux loss: {aux_loss.item():.4f}")
    assert output.shape == x.shape
    print("   [PASS]")
    return True


def test_mamba():
    """Test State Space Model (Mamba)"""
    print("\n" + "=" * 60)
    print("Testing Mamba (State Space Model)")
    print("=" * 60)

    from nicto_ai.core.mamba import MambaStack

    dim = 256
    mamba = MambaStack(n_layers=4, d_model=dim)
    x = torch.randn(2, 16, dim)
    output, caches = mamba(x)

    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    assert output.shape == x.shape
    print("   [PASS]")
    return True


def test_liquid():
    """Test Liquid Neural Network"""
    print("\n" + "=" * 60)
    print("Testing Liquid Neural Network")
    print("=" * 60)

    from nicto_ai.core.liquid import LiquidStack

    dim = 256
    liquid = LiquidStack(n_layers=4, dim=dim, n_neurons=128)
    x = torch.randn(2, 16, dim)
    output, states = liquid(x)

    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    assert output.shape == x.shape
    print("   [PASS]")
    return True


def test_consciousness():
    """Test Consciousness Layer"""
    print("\n" + "=" * 60)
    print("Testing Consciousness Layer")
    print("=" * 60)

    from nicto_ai.core.consciousness import ConsciousnessLayer

    dim = 256
    consciousness = ConsciousnessLayer(dim=dim)
    x = torch.randn(2, 16, dim)
    outputs = consciousness(x)

    print(f"   Input: {x.shape}")
    print(f"   Output: {outputs['output'].shape}")
    print(f"   Self-model keys: {list(outputs['self_model'].keys())}")
    print(f"   Intentions: {list(outputs['intentions'].keys())}")
    assert outputs['output'].shape[0] == 2
    print("   [PASS]")
    return True


def test_emotion():
    """Test Emotional Processing"""
    print("\n" + "=" * 60)
    print("Testing Emotional Processing")
    print("=" * 60)

    from nicto_ai.core.emotion import EmotionalResponseGenerator

    dim = 256
    emotion = EmotionalResponseGenerator(dim)
    x = torch.randn(2, 16, dim)
    outputs = emotion(x)

    print(f"   Input: {x.shape}")
    print(f"   Output: {outputs['output'].shape}")
    print(f"   Emotion: {outputs['emotion'].shape}")
    assert outputs['output'].shape[0] == 2
    print("   [PASS]")
    return True


def test_memory():
    """Test Hierarchical Memory"""
    print("\n" + "=" * 60)
    print("Testing Hierarchical Memory")
    print("=" * 60)

    from nicto_ai.core.memory import HierarchicalMemory

    dim = 256
    memory = HierarchicalMemory(dim=dim)
    x = torch.randn(2, 16, dim)
    outputs = memory(x, access_pattern="full")

    print(f"   Input: {x.shape}")
    print(f"   Fused: {outputs['fused'].shape}")
    assert outputs['fused'].shape == x.shape
    print("   [PASS]")
    return True


def test_full_model():
    """Test complete NICTO model"""
    print("\n" + "=" * 60)
    print("Testing Complete NICTO Model")
    print("=" * 60)

    from nicto_ai.core.model import NICTOModel
    from nicto_ai.core.vision import VGG16ForNICTO

    print("\nCreating small model...")
    model = NICTOModel(
        vocab_size=32000,
        dim=1024,
        max_seq_len=2048,
    )
    model.vgg16 = VGG16ForNICTO(nicto_dim=1024, pretrained=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Test text forward
    print("\nTesting text forward pass...")
    input_ids = torch.randint(0, 32000, (2, 32))
    outputs = model(input_ids)
    print(f"   Input: {input_ids.shape}")
    print(f"   Logits: {outputs['logits'].shape}")
    assert outputs['logits'].shape == (2, 32, 32000)
    print("   [PASS]")

    # Test with labels
    print("\nTesting with labels (loss computation)...")
    labels = torch.randint(0, 32000, (2, 32))
    outputs = model(input_ids, labels=labels)
    print(f"   Loss: {outputs['loss'].item():.4f}")
    assert outputs['loss'] is not None
    print("   [PASS]")

    # Test image forward
    print("\nTesting image forward pass...")
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        vision_outputs = model.vgg16(images)
    print(f"   Vision features: {vision_outputs['global_features'].shape}")
    print("   [PASS]")

    # Test generation
    print("\nTesting text generation...")
    prompt = torch.randint(0, 32000, (1, 10))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    print(f"   Prompt: {prompt.shape}")
    print(f"   Generated: {generated.shape}")
    assert generated.shape[1] > 10
    print("   [PASS]")

    print("\n" + "=" * 60)
    print("All model tests passed!")
    print("=" * 60)
    return True


def test_training_pipeline():
    """Test training pipeline"""
    print("\n" + "=" * 60)
    print("Testing Training Pipeline")
    print("=" * 60)

    from nicto_ai.core.model import NICTOModel
    from nicto_ai.training.trainer import NICTOTrainer
    from torch.utils.data import TensorDataset, DataLoader

    # Create tiny model
    model = NICTOModel(vocab_size=1000, dim=256, max_seq_len=128)

    # Create dummy data
    data = torch.randint(0, 1000, (100, 128))
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Training config
    config = {
        "learning_rate": 1e-4,
        "min_lr": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 5,
        "total_steps": 20,
    }

    # Create trainer
    trainer = NICTOTrainer(
        model=model,
        config=config,
        output_dir="test_checkpoints",
        device="cpu",
    )

    # Run a few training steps
    print("\nRunning 5 training steps...")
    model.train()
    for step, (batch,) in enumerate(loader):
        if step >= 5:
            break

        input_ids = batch
        labels = batch.clone()

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        print(f"   Step {step}: Loss = {loss.item():.4f}")

    print("   [PASS]")
    print("\nTraining pipeline test complete!")
    return True


if __name__ == "__main__":
    print("NICTO AI - Complete Test Suite")
    print("=" * 60)

    results = []
    results.append(("VGG16", test_vgg16()))
    results.append(("MLA", test_mla()))
    results.append(("MoE", test_moe()))
    results.append(("Mamba", test_mamba()))
    results.append(("Liquid", test_liquid()))
    results.append(("Consciousness", test_consciousness()))
    results.append(("Emotion", test_emotion()))
    results.append(("Memory", test_memory()))
    results.append(("Full Model", test_full_model()))
    results.append(("Training", test_training_pipeline()))

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
