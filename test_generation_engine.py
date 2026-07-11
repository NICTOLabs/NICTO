"""
Test Generation Engine.

End-to-end tests for all generation modules.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nicto_ai.nictos.generation.unified_vae import UnifiedVAE
from nicto_ai.nictos.generation.flow_matching import DiT, FlowMatchingTrainer, FlowMatchingSampler
from nicto_ai.nictos.generation.video_generator import VideoGenerator
from nicto_ai.nictos.generation.audio_generator import MusicGenerator, SFXGenerator
from nicto_ai.nictos.generation.native_interleaving import NativeInterleaving, InterleavedSequence, Modality
from nicto_ai.nictos.generation.creativity_engine import CreativityEngine


def test_unified_vae():
    """Test Unified VAE."""
    print("Testing Unified VAE...")

    vae = UnifiedVAE(latent_dim=512, in_channels=3)

    # Test image encoding
    images = torch.randn(2, 3, 64, 64)
    mean, logvar = vae.encode(images, "image")
    z = vae.reparameterize(mean, logvar)
    recon = vae.decode(z, "image")
    print(f"  Image: {images.shape} -> {z.shape} -> {recon.shape}")

    # Test video encoding
    video = torch.randn(2, 3, 16, 64, 64)
    mean_v, logvar_v = vae.encode(video, "video")
    z_v = vae.reparameterize(mean_v, logvar_v)
    recon_v = vae.decode(z_v, "video")
    print(f"  Video: {video.shape} -> {z_v.shape} -> {recon_v.shape}")

    # Test audio encoding
    audio = torch.randn(2, 1, 16000)
    mean_a, logvar_a = vae.encode(audio, "audio")
    z_a = vae.reparameterize(mean_a, logvar_a)
    recon_a = vae.decode(z_a, "audio")
    print(f"  Audio: {audio.shape} -> {z_a.shape} -> {recon_a.shape}")

    # Test text encoding
    text = torch.randint(0, 30000, (2, 100))
    mean_t, logvar_t = vae.encode(text, "text")
    print(f"  Text: {text.shape} -> {mean_t.shape}")

    # Test losses per modality
    loss_i = vae.loss(images, recon, mean, logvar)
    print(f"  Image loss: {loss_i.item():.4f}")
    loss_v = vae.loss(video, recon_v, mean_v, logvar_v)
    print(f"  Video loss: {loss_v.item():.4f}")
    loss_a = vae.loss(audio, recon_a, mean_a, logvar_a)
    print(f"  Audio loss: {loss_a.item():.4f}")

    # Test sampling
    samples = vae.sample(2, "image")
    print(f"  Samples: {samples.shape}")

    print("  [OK] Unified VAE passed\n")
    return vae


def test_flow_matching(vae):
    """Test Flow Matching DiT."""
    print("Testing Flow Matching DiT...")

    dit = DiT(
        hidden_dim=256,
        context_dim=512,
        num_layers=4,
        num_heads=4,
    )

    # Test forward pass
    x_t = torch.randn(2, 512, 8, 8)
    t = torch.rand(2)
    context = torch.randn(2, 100, 512)

    v = dit(x_t, t, context)
    print(f"  Forward: {x_t.shape} -> {v.shape}")

    # Test trainer
    trainer = FlowMatchingTrainer(dit, vae)
    images = torch.randn(2, 3, 64, 64)
    loss = trainer.compute_loss(images, context, "image")
    print(f"  Loss: {loss.item():.4f}")

    # Test sampler
    sampler = FlowMatchingSampler(dit, vae, num_steps=5)
    generated = sampler.sample(context, "image", (2, 512, 8, 8))
    print(f"  Generated: {generated.shape}")

    print("  [OK] Flow Matching DiT passed\n")


def test_video_generator():
    """Test Video Generator."""
    print("Testing Video Generator...")

    generator = VideoGenerator(
        latent_dim=256,
        hidden_dim=256,
        context_dim=512,
        num_layers=4,
        num_heads=4,
    )

    # Test generation
    context = torch.randn(1, 10, 512)
    video = generator.generate(
        context,
        num_frames=16,  # Short for testing
        height=32,
        width=32,
        num_steps=2,
    )
    print(f"  Generated video: {video.shape}")

    # Test AV sync loss
    video_features = torch.randn(1, 512, 2, 4, 4)
    audio_features = torch.randn(1, 512, 8)
    loss = generator.compute_av_sync_loss(video_features, audio_features)
    print(f"  AV Sync loss: {loss.item():.4f}")

    print("  [OK] Video Generator passed\n")


def test_audio_generator():
    """Test Audio Generator."""
    print("Testing Audio Generator...")

    # Test Music Generator
    music_gen = MusicGenerator(
        dim=256,
        context_dim=512,
        codebook_size=512,
        num_codebooks=4,
        num_layers=4,
        num_heads=4,
    )

    context = torch.randn(1, 10, 512)
    audio = music_gen.generate(
        context,
        duration=0.5,  # Short for testing
        num_steps=2,
    )
    print(f"  Generated music: {audio.shape}")

    # Test SFX Generator
    sfx_gen = SFXGenerator(dim=256, context_dim=512, num_layers=4)
    audio = sfx_gen.generate(context, duration=0.5, num_steps=2)
    print(f"  Generated SFX: {audio.shape}")

    print("  [OK] Audio Generator passed\n")


def test_native_interleaving():
    """Test Native Interleaving."""
    print("Testing Native Interleaving...")

    generator = NativeInterleaving(
        vocab_size=30000,
        image_vocab_size=512,
        video_vocab_size=512,
        audio_vocab_size=512,
        embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=4,
    )

    # Test forward pass
    tokens = torch.randint(0, 30000, (1, 10))
    logits = generator(tokens, Modality.TEXT)
    print(f"  Text logits: {logits.shape}")

    tokens = torch.randint(0, 512, (1, 64))
    logits = generator(tokens, Modality.IMAGE)
    print(f"  Image logits: {logits.shape}")

    # Test generation
    seq = generator.generate(
        "Generate a cat",
        modalities=["text", "image"],
        max_length=10,
    )
    print(f"  Generated {len(seq.segments)} segments")

    print("  [OK] Native Interleaving passed\n")


def test_creativity_engine():
    """Test Creativity Engine."""
    print("Testing Creativity Engine...")

    engine = CreativityEngine(latent_dim=512, hidden_dim=1024)

    # Test consistency check
    z1 = torch.randn(2, 512)
    z2 = torch.randn(2, 512)
    score = engine.check_consistency(z1, z2, "image", "video")
    print(f"  Consistency score: {score.shape}")

    # Test style transfer
    z_content = torch.randn(2, 512)
    z_style = torch.randn(2, 512)
    z_stylized = engine.transfer_style(z_content, z_style)
    print(f"  Stylized: {z_stylized.shape}")

    # Test concept blending
    z1 = torch.randn(2, 512)
    z2 = torch.randn(2, 512)
    z_blended = engine.blend_concepts(z1, z2, "text", "audio", 0.5)
    print(f"  Blended: {z_blended.shape}")

    # Test in-context generation
    z_ref = torch.randn(2, 512)
    z_new = engine.generate_consistent(z_ref, "video", "image")
    print(f"  In-context: {z_new.shape}")

    # Test GAN validate and inspire
    z_created = torch.randn(2, 512)
    z_target = torch.randn(2, 512)
    z_inspired, score = engine.validate_and_inspire(z_created, z_target)
    print(f"  Validate+Inspire: score={score.mean().item():.3f}, inspired={z_inspired.shape}")

    # Test iterate refinement loop
    def dummy_generator(z):
        return torch.randn(z.shape[0], 3, 64, 64)

    initial = torch.randn(2, 512)
    best_output, scores = engine.iterate_refinement(
        dummy_generator, initial, z_target, max_iterations=3
    )
    print(f"  Iterate refinement: {len(scores)} steps, scores={[f'{s:.3f}' for s in scores]}")

    print("  [OK] Creativity Engine passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("NICTO Generation Engine Tests")
    print("=" * 60 + "\n")

    try:
        vae = test_unified_vae()
        test_flow_matching(vae)
        test_video_generator()
        test_audio_generator()
        test_native_interleaving()
        test_creativity_engine()

        print("=" * 60)
        print("All tests passed! [OK]")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
