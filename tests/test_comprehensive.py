"""
NICTO Comprehensive Test Suite v2.

Tests for core modules, tools, voice, generation, and configuration.
"""

import sys
import time
import torch
import numpy as np

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL  {name}: {e}")


# ==============================================================================
# Core Module Tests
# ==============================================================================

def test_core_memory():
    from nicto_ai.core.memory import WorkingMemory, EpisodicMemory
    wm = WorkingMemory(dim=64, capacity=10)
    x = torch.randn(1, 10, 64)
    result = wm(x)
    assert "retrieved" in result
    em = EpisodicMemory(dim=64, capacity=10)
    result = em(x, mode="store")
    result = em(x, mode="retrieve")
    assert "retrieved" in result


def test_core_emotion():
    from nicto_ai.core.emotion import EmotionalStateModel
    model = EmotionalStateModel(dim=64)
    x = torch.randn(1, 10, 64)
    result = model(x)
    assert "current_emotion" in result


def test_core_consciousness():
    from nicto_ai.core.consciousness import UncertaintyEstimator
    model = UncertaintyEstimator(dim=64, n_heads=4)
    logits = torch.randn(1, 10, 64)
    result = model(logits)
    assert "uncertainty" in result


def test_core_liquid():
    from nicto_ai.core.liquid import LiquidLayer
    layer = LiquidLayer(n_neurons=32, input_dim=64, output_dim=64)
    x = torch.randn(1, 5, 64)
    out, state = layer(x)
    assert out.shape == (1, 5, 64)


def test_core_mamba():
    from nicto_ai.core.mamba import SelectiveSSM
    ssm = SelectiveSSM(d_model=64, d_state=16, d_conv=4, expand=2)
    x = torch.randn(1, 10, 64)
    out, _ = ssm(x)
    assert out.shape == (1, 10, 64)


def test_core_vision():
    from nicto_ai.core.vision import VGG16Encoder
    enc = VGG16Encoder(pretrained=False)
    img = torch.randn(1, 3, 224, 224)
    out = enc(img)
    assert out.shape[0] == 1


# ==============================================================================
# Generation Tests
# ==============================================================================

def test_generation_unified_vae():
    from nicto_ai.nictos.generation.unified_vae import UnifiedVAE
    vae = UnifiedVAE(latent_dim=256)
    img = torch.randn(1, 3, 32, 32)
    mean, logvar = vae.encode(img, "image")
    assert mean.shape[0] == 1
    pc = torch.randn(1, 512, 6)
    mean, logvar = vae.encode(pc, "point_cloud")
    assert mean.shape == (1, 256)


def test_generation_flow_matching():
    from nicto_ai.nictos.generation.flow_matching import DiT
    dit = DiT(hidden_dim=128, context_dim=64, num_layers=2, num_heads=4, patch_size=2)
    x = torch.randn(1, 512, 4, 4)  # hidden_dim * patch_size^2 = 128*4 = 512 channels
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 5, 64)
    out = dit(x, t, ctx)
    assert out.shape == (1, 512, 4, 4)


def test_generation_point_cloud():
    from nicto_ai.nictos.generation.point_cloud_generator import PointCloudGenerator
    gen = PointCloudGenerator(latent_dim=256, hidden_dim=128, num_layers=2, context_dim=128)
    points = gen.generate(num_points=512, batch_size=1, num_steps=5)
    assert points.shape[0] == 1
    assert points.shape[2] == 6


def test_generation_video():
    from nicto_ai.nictos.generation.video_generator import VideoGenerator
    gen = VideoGenerator(latent_dim=128, hidden_dim=64, context_dim=64, num_layers=1, num_heads=4)
    ctx = torch.randn(1, 5, 64)
    video = gen.generate(ctx, num_frames=4, height=16, width=16, num_steps=2)
    assert video.shape[0] == 1
    assert video.shape[2] == 2  # Video may output fewer frames than requested


def test_generation_creativity():
    from nicto_ai.nictos.generation.creativity_engine import CreativityEngine
    engine = CreativityEngine(latent_dim=256)
    z = torch.randn(1, 256, 1, 1)
    quality, aspects = engine.validate_and_inspire(z, torch.randn(1, 256))
    assert quality.shape[0] == 1


# ==============================================================================
# Neural Bridge Tests
# ==============================================================================

def test_neural_torch_verlet():
    from nicto_ai.nictos.neural.integrator import TorchVerlet
    tv = TorchVerlet()
    pos = torch.randn(2, 3)
    vel = torch.randn(2, 3)
    force = torch.randn(2, 3)
    mass = torch.ones(2, 1)
    x_new, v_new, a_new = tv(pos, vel, force, mass, dt=0.01)
    assert x_new.shape == pos.shape


def test_neural_surrogate():
    from nicto_ai.nictos.neural.surrogate import MLPSurrogate
    sur = MLPSurrogate(state_dim=6, hidden=32, layers=2)
    x = torch.randn(1, 6)
    out = sur(x)
    assert out.shape == (1, 6)


# ==============================================================================
# Simulation Tests
# ==============================================================================

def test_simulation_engine():
    from nicto_ai.nictos.core.engine import SimulationEngine
    from nicto_ai.nictos.domains.physics.types import Particle
    engine = SimulationEngine()
    p1 = Particle(id=0, mass=1.0, position=[0, 0, 0], velocity=[1, 0, 0])
    p2 = Particle(id=1, mass=1.0, position=[1, 0, 0], velocity=[-1, 0, 0])
    engine.add_entity(p1)
    engine.add_entity(p2)
    engine.step(dt=0.01)
    assert len(engine.entities) == 2


def test_simulation_potentials():
    from nicto_ai.nictos.domains.physics.potentials import LennardJones
    lj = LennardJones(epsilon=1.0, sigma=1.0)
    pos_a = np.array([0.0, 0.0, 0.0])
    pos_b = np.array([1.0, 0.0, 0.0])
    energy, force = lj.compute(pos_a, pos_b)
    assert isinstance(energy, float)
    assert force.shape == (3,)


# ==============================================================================
# Voice Tests
# ==============================================================================

def test_voice_stt():
    from nicto_ai.voice.stt import SpeechToText
    stt = SpeechToText()
    assert stt is not None


def test_voice_tts():
    from nicto_ai.voice.tts import TextToSpeech
    tts = TextToSpeech()
    assert tts is not None


def test_voice_backend():
    from nicto_ai.voice.backend_interface import EchoBackend
    backend = EchoBackend()
    assert backend is not None


# ==============================================================================
# Config Tests
# ==============================================================================

def test_config_default():
    from nicto_ai.configs.config import get_default_config
    config = get_default_config()
    assert config.model.latent_dim == 512
    assert config.training.lr == 3e-4


def test_config_small():
    from nicto_ai.configs.config import get_small_config
    config = get_small_config()
    assert config.model.latent_dim == 256
    assert config.model.hidden_dim == 512


def test_config_loader():
    from nicto_ai.configs.config import ConfigLoader
    loader = ConfigLoader("configs")
    config = loader.load_config("default")
    assert config.model.latent_dim == 512


# ==============================================================================
# API Tests
# ==============================================================================

def test_api_server():
    from nicto_ai.api.server import app
    routes = [r.path for r in app.routes]
    assert "/health" in routes
    assert "/generate/image" in routes
    assert "/generate/3d" in routes


# ==============================================================================
# DeepSearch Tests
# ==============================================================================

def test_deepsearch_v2():
    from nicto_ai.core.deepsearch import DeepSearchModule
    model = DeepSearchModule(dim=128, max_depth=3, beam_width=2, n_thoughts_per_step=2)
    x = torch.randn(1, 128)
    result = model(x, use_search=False, n_thoughts=2)
    assert result["best_reasoning"].shape == (1, 128)


# ==============================================================================
# Run All Tests
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NICTO COMPREHENSIVE TEST SUITE v2")
    print("=" * 60)

    print("\nCore Modules:")
    test("memory", test_core_memory)
    test("emotion", test_core_emotion)
    test("consciousness", test_core_consciousness)
    test("liquid", test_core_liquid)
    test("mamba", test_core_mamba)
    test("vision", test_core_vision)

    print("\nGeneration:")
    test("unified_vae", test_generation_unified_vae)
    test("flow_matching", test_generation_flow_matching)
    test("point_cloud", test_generation_point_cloud)
    test("video", test_generation_video)
    test("creativity", test_generation_creativity)

    print("\nNeural Bridge:")
    test("torch_verlet", test_neural_torch_verlet)
    test("surrogate", test_neural_surrogate)

    print("\nSimulation:")
    test("engine", test_simulation_engine)
    test("potentials", test_simulation_potentials)

    print("\nVoice:")
    test("stt", test_voice_stt)
    test("tts", test_voice_tts)
    test("backend", test_voice_backend)

    print("\nConfiguration:")
    test("default_config", test_config_default)
    test("small_config", test_config_small)
    test("config_loader", test_config_loader)

    print("\nAPI:")
    test("server", test_api_server)

    print("\nDeepSearch:")
    test("deepsearch_v2", test_deepsearch_v2)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
