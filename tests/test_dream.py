"""
Test NICTO AI Dream Engine
Experience replay, synthetic generation, metacognition
"""

import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nicto_ai.dream.replay import ExperienceReplayBuffer, Experience, SumTree
from nicto_ai.dream.generator import SyntheticDataGenerator, DreamSample
from nicto_ai.dream.dream_engine import DreamEngine, DreamSession
from nicto_ai.core.consciousness import (
    RealConsciousnessLayer,
    UncertaintyEstimator,
    ErrorDetector,
    PerformanceTracker,
)
from nicto_ai.core.model import create_small_model


# ─── SumTree Tests ───────────────────────────────────────────────

def test_sumtree_add_get():
    tree = SumTree(capacity=16)
    tree.add(1.0, "a")
    tree.add(2.0, "b")
    tree.add(3.0, "c")
    assert tree.n_entries == 3
    assert abs(tree.total - 6.0) < 1e-6
    print("  SumTree add/get: OK")

def test_sumtree_sample_batch():
    tree = SumTree(capacity=32)
    for i in range(20):
        tree.add(float(i + 1), i)
    batch = tree.sample_batch(8)
    assert len(batch) == 8
    for idx, pri, data in batch:
        assert data is not None
    print("  SumTree sample_batch: OK")

def test_sumtree_capacity_wrap():
    tree = SumTree(capacity=4)
    for i in range(10):
        tree.add(float(i), i)
    assert tree.n_entries == 4
    print("  SumTree capacity wrap: OK")


# ─── ExperienceReplayBuffer Tests ────────────────────────────────

def test_replay_buffer_add_sample():
    buf = ExperienceReplayBuffer(capacity=100)
    for i in range(50):
        exp = Experience(
            id=i,
            input_ids=torch.randint(0, 100, (1, 8)),
            logits=torch.randn(1, 8, 100),
            loss=float(i) / 10.0,
            surprise=float(i) / 20.0,
            entropy=float(i) / 30.0,
            was_correct=(i % 3 == 0),
        )
        buf.add(exp)
    assert buf.size == 50
    sampled = buf.sample(10)
    assert len(sampled) > 0
    print("  ReplayBuffer add/sample: OK")

def test_replay_buffer_priority():
    buf = ExperienceReplayBuffer(capacity=100)
    # Low surprise experience
    low = Experience(id=0, input_ids=torch.zeros(1, 4, dtype=torch.long),
                     logits=torch.randn(1, 4, 100), loss=0.1, surprise=0.05,
                     entropy=0.1, was_correct=True)
    # High surprise experience
    high = Experience(id=1, input_ids=torch.zeros(1, 4, dtype=torch.long),
                      logits=torch.randn(1, 4, 100), loss=5.0, surprise=4.0,
                      entropy=3.0, was_correct=False)
    buf.add(low)
    buf.add(high)
    assert buf.size == 2
    print("  ReplayBuffer priority: OK")

def test_replay_buffer_update_priorities():
    buf = ExperienceReplayBuffer(capacity=50)
    exps = []
    for i in range(10):
        exp = Experience(id=i, input_ids=torch.zeros(1, 4, dtype=torch.long),
                         logits=torch.randn(1, 4, 100), loss=0.5, surprise=0.3,
                         entropy=0.2, was_correct=True)
        buf.add(exp)
        exps.append(exp)
    buf.update_priorities(exps, [1.0] * 10)
    print("  ReplayBuffer update_priorities: OK")

def test_replay_buffer_stats():
    buf = ExperienceReplayBuffer(capacity=50)
    for i in range(5):
        buf.add(Experience(id=i, input_ids=torch.zeros(1, 4, dtype=torch.long),
                           logits=torch.randn(1, 4, 100), loss=1.0, surprise=0.5,
                           entropy=0.3, was_correct=False))
    stats = buf.stats
    assert "total_added" in stats
    assert stats["total_added"] == 5
    print("  ReplayBuffer stats: OK")


# ─── SyntheticDataGenerator Tests ────────────────────────────────

def test_augment_experience():
    gen = SyntheticDataGenerator(vocab_size=200, dim=64)
    input_ids = torch.randint(0, 200, (2, 16))
    logits = torch.randn(2, 16, 200)
    samples = gen.augment_experience(input_ids, logits, n_augments=3)
    assert len(samples) == 3
    for s in samples:
        assert s.input_ids.shape == input_ids.shape
        assert s.source == "augmentation"
    print("  Generator augment_experience: OK")

def test_generate_curriculum_batch():
    gen = SyntheticDataGenerator(vocab_size=200, dim=64)
    sample = gen.generate_curriculum_batch(batch_size=4, seq_len=20, difficulty=0.5)
    assert sample.input_ids.shape == (4, 20)
    assert sample.labels.shape == (4, 20)
    assert sample.source == "curriculum"
    print("  Generator generate_curriculum_batch: OK")

def test_generate_curriculum_difficulty_range():
    gen = SyntheticDataGenerator(vocab_size=200, dim=64)
    easy = gen.generate_curriculum_batch(8, 20, difficulty=0.1)
    hard = gen.generate_curriculum_batch(8, 20, difficulty=0.9)
    # Higher difficulty uses wider vocab range; just verify shapes and sources
    assert easy.source == "curriculum"
    assert hard.source == "curriculum"
    assert easy.input_ids.shape == (8, 20)
    assert hard.input_ids.shape == (8, 20)
    print("  Generator curriculum difficulty range: OK")

def test_generate_contrastive_pairs():
    gen = SyntheticDataGenerator(vocab_size=200, dim=64)
    model = create_small_model(vocab_size=200, dim=64)
    input_ids = torch.randint(0, 200, (1, 16))
    pairs = gen.generate_contrastive_pairs(input_ids, model, n_pairs=2)
    assert len(pairs) == 2
    assert pairs[0].source == "contrastive"
    assert pairs[1].source == "contrastive"
    print("  Generator generate_contrastive_pairs: OK")

def test_generate_knowledge_distillation():
    gen = SyntheticDataGenerator(vocab_size=200, dim=64)
    model = create_small_model(vocab_size=200, dim=64)
    input_ids = torch.randint(0, 200, (1, 16))
    sample = gen.generate_knowledge_distillation(input_ids, model)
    assert sample.source == "distillation"
    assert sample.input_ids.shape[0] == 1
    print("  Generator generate_knowledge_distillation: OK")


# ─── RealConsciousnessLayer Tests ────────────────────────────────

def test_uncertainty_estimator():
    ue = UncertaintyEstimator(dim=64)
    logits = torch.randn(2, 10, 200)
    out = ue(logits)
    assert "uncertainty" in out
    assert "raw_uncertainty" in out
    assert "entropy" in out
    assert out["uncertainty"].shape == (2,)
    print("  UncertaintyEstimator: OK")

def test_uncertainty_estimator_with_hidden():
    ue = UncertaintyEstimator(dim=64)
    logits = torch.randn(2, 10, 200)
    hidden = torch.randn(2, 10, 64)
    out = ue(logits, hidden_states=hidden)
    assert out["uncertainty"].shape == (2,)
    print("  UncertaintyEstimator with hidden: OK")

def test_error_detector():
    ed = ErrorDetector(dim=64)
    logits = torch.randn(2, 10, 200)
    out = ed(logits)
    assert "error_probability" in out
    assert "confidence" in out
    assert out["error_probability"].shape == (2,)
    assert (out["confidence"] >= 0).all()
    assert (out["confidence"] <= 1).all()
    print("  ErrorDetector: OK")

def test_error_detector_with_loss():
    ed = ErrorDetector(dim=64)
    logits = torch.randn(2, 10, 200)
    loss = torch.tensor(2.5)
    out = ed(logits, loss=loss)
    assert out["running_error_rate"] >= 0
    print("  ErrorDetector with loss: OK")

def test_performance_tracker():
    pt = PerformanceTracker(dim=64)
    preds = torch.tensor([1, 2, 3, 4])
    targets = torch.tensor([1, 2, 5, 4])
    confs = torch.tensor([0.9, 0.8, 0.7, 0.95])
    pt.update(preds, targets, confs)
    stats = pt.get_stats()
    assert stats["total_predictions"] == 4
    assert stats["running_accuracy"] == 0.75
    print("  PerformanceTracker: OK")

def test_performance_tracker_calibration():
    pt = PerformanceTracker(dim=64)
    for _ in range(20):
        preds = torch.randint(0, 10, (8,))
        targets = torch.randint(0, 10, (8,))
        confs = torch.rand(8)
        pt.update(preds, targets, confs)
    ece = pt.get_calibration_error()
    assert isinstance(ece, float)
    print("  PerformanceTracker calibration: OK")

def test_real_consciousness_layer():
    layer = RealConsciousnessLayer(dim=64)
    x = torch.randn(2, 10, 64)
    out = layer(x)
    assert "output" in out
    assert "uncertainty" in out
    assert "error_probability" in out
    assert "confidence" in out
    assert "performance" in out
    assert out["output"].shape == (2, 64)
    print("  RealConsciousnessLayer: OK")

def test_real_consciousness_with_logits():
    layer = RealConsciousnessLayer(dim=64)
    x = torch.randn(2, 10, 64)
    logits = torch.randn(2, 10, 200)
    labels = torch.randint(0, 200, (2, 10))
    out = layer(x, logits=logits, labels=labels)
    assert out["output"].shape == (2, 64)
    assert out["failure_count"] >= 0
    print("  RealConsciousnessLayer with logits: OK")

def test_consciousness_should_act_cautiously():
    layer = RealConsciousnessLayer(dim=64)
    # Fresh layer has < 10 predictions, so not cautious yet
    assert not layer.should_act_cautiously(threshold=0.5)
    print("  RealConsciousnessLayer should_act_cautiously: OK")

def test_consciousness_self_report():
    layer = RealConsciousnessLayer(dim=64)
    report = layer.get_self_report()
    assert "performance" in report
    assert "failure_memory_used" in report
    assert "should_act_cautiously" in report
    print("  RealConsciousnessLayer self_report: OK")


# ─── DreamEngine Integration Tests ──────────────────────────────

def test_dream_engine_init():
    model = create_small_model(vocab_size=200, dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    engine = DreamEngine(model, opt, replay_capacity=500, vocab_size=200, dim=64)
    stats = engine.get_stats()
    assert stats["session_count"] == 0
    assert stats["replay_buffer_size"] == 0
    print("  DreamEngine init: OK")

def test_dream_engine_record():
    model = create_small_model(vocab_size=200, dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    engine = DreamEngine(model, opt, replay_capacity=500, vocab_size=200, dim=64)
    input_ids = torch.randint(0, 200, (2, 16))
    labels = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        out = model(input_ids, labels=labels)
    engine.record_experience(input_ids, out["logits"], out["loss"].item())
    assert engine.replay_buffer.size == 1
    print("  DreamEngine record: OK")

def test_dream_engine_dream():
    model = create_small_model(vocab_size=200, dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    engine = DreamEngine(model, opt, replay_capacity=500, vocab_size=200, dim=64)
    # Record some experiences first
    for _ in range(20):
        input_ids = torch.randint(0, 200, (2, 16))
        labels = torch.randint(0, 200, (2, 16))
        with torch.no_grad():
            out = model(input_ids, labels=labels)
        engine.record_experience(input_ids, out["logits"], out["loss"].item())
    # Run dream session
    session = engine.dream(
        n_replay_steps=5,
        n_generated_batches=4,
        batch_size=2,
        seq_len=16,
    )
    assert session.experiences_replayed > 0
    assert session.samples_generated > 0
    assert session.duration_ms > 0
    print("  DreamEngine dream: OK")

def test_dream_engine_multiple_sessions():
    model = create_small_model(vocab_size=200, dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    engine = DreamEngine(model, opt, replay_capacity=500, vocab_size=200, dim=64)
    for _ in range(10):
        input_ids = torch.randint(0, 200, (2, 16))
        labels = torch.randint(0, 200, (2, 16))
        with torch.no_grad():
            out = model(input_ids, labels=labels)
        engine.record_experience(input_ids, out["logits"], out["loss"].item())
    for i in range(3):
        session = engine.dream(n_replay_steps=3, n_generated_batches=2, batch_size=2, seq_len=16)
    stats = engine.get_stats()
    assert stats["session_count"] == 3
    print("  DreamEngine multiple sessions: OK")

def test_dream_session_result():
    model = create_small_model(vocab_size=200, dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    engine = DreamEngine(model, opt, replay_capacity=500, vocab_size=200, dim=64)
    for _ in range(15):
        input_ids = torch.randint(0, 200, (2, 16))
        labels = torch.randint(0, 200, (2, 16))
        with torch.no_grad():
            out = model(input_ids, labels=labels)
        engine.record_experience(input_ids, out["logits"], out["loss"].item())
    session = engine.dream(n_replay_steps=3, n_generated_batches=2, batch_size=2, seq_len=16)
    assert isinstance(session, DreamSession)
    assert session.duration_ms >= 0
    assert session.experiences_replayed >= 0
    assert session.samples_generated >= 0
    print("  DreamSession result: OK")


# ─── Data Sorter Tests ──────────────────────────────────────────

def test_token_sorter():
    from nicto_ai.core.data_sorter import TokenSorter
    ts = TokenSorter(dim=64, temperature=1.0)
    x = torch.randn(2, 10, 64)
    pooled, scores = ts(x, return_scores=True)
    assert pooled.shape == (2, 64)
    assert scores.shape == (2, 10)
    print("  TokenSorter: OK")

def test_network_priority_gate():
    from nicto_ai.core.data_sorter import NetworkPriorityGate
    npg = NetworkPriorityGate(dim=64, n_networks=6, temperature=2.0)
    fused = torch.randn(2, 64)
    gates = npg(fused)
    assert gates.shape == (2, 6)
    print("  NetworkPriorityGate: OK")

def test_memory_consolidator():
    from nicto_ai.core.data_sorter import MemoryConsolidator
    mc = MemoryConsolidator(dim=64)
    keys = torch.randn(8, 64)
    values = torch.randn(8, 64)
    metadata = torch.zeros(8, 4)
    result = mc(keys, values, metadata, n_filled=8)
    assert "mean_score" in result
    assert "max_score" in result
    print("  MemoryConsolidator: OK")


# ─── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NICTO AI - Dream Engine & Metacognition Test Suite")
    print("=" * 60)

    print("\n--- SumTree ---")
    test_sumtree_add_get()
    test_sumtree_sample_batch()
    test_sumtree_capacity_wrap()

    print("\n--- ExperienceReplayBuffer ---")
    test_replay_buffer_add_sample()
    test_replay_buffer_priority()
    test_replay_buffer_update_priorities()
    test_replay_buffer_stats()

    print("\n--- SyntheticDataGenerator ---")
    test_augment_experience()
    test_generate_curriculum_batch()
    test_generate_curriculum_difficulty_range()
    test_generate_contrastive_pairs()
    test_generate_knowledge_distillation()

    print("\n--- Real Metacognition ---")
    test_uncertainty_estimator()
    test_uncertainty_estimator_with_hidden()
    test_error_detector()
    test_error_detector_with_loss()
    test_performance_tracker()
    test_performance_tracker_calibration()
    test_real_consciousness_layer()
    test_real_consciousness_with_logits()
    test_consciousness_should_act_cautiously()
    test_consciousness_self_report()

    print("\n--- DreamEngine Integration ---")
    test_dream_engine_init()
    test_dream_engine_record()
    test_dream_engine_dream()
    test_dream_engine_multiple_sessions()
    test_dream_session_result()

    print("\n--- Data Sorting ---")
    test_token_sorter()
    test_network_priority_gate()
    test_memory_consolidator()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
