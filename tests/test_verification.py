"""
Test NICTO AI Anti-Hallucination Engine
Claim extraction, verification, grounding, confidence, refusal
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nicto_ai.verification.claim_extractor import ClaimExtractor, Claim, ClaimType, ClaimSeverity
from nicto_ai.verification.verifier import ClaimVerifier, VerificationResult, VerificationStatus
from nicto_ai.verification.grounding import GroundingLayer, GroundingResult
from nicto_ai.verification.confidence import ConfidenceScorer, ConfidenceScore
from nicto_ai.verification.refusal import RefusalGate, RefusalDecision, RefusalReason
from nicto_ai.verification.attribution import SourceAttribution, Source, Attribution
from nicto_ai.verification.engine import AntiHallucinationEngine, VerificationReport


# ─── ClaimExtractor Tests ───────────────────────────────────────

def test_extract_basic_claims():
    ext = ClaimExtractor()
    text = "Python is a programming language. It was created in 1991."
    claims = ext.extract(text)
    assert len(claims) > 0
    print("  Extract basic claims: OK")

def test_extract_factual_claims():
    ext = ClaimExtractor()
    text = "The Earth orbits the Sun. Water boils at 100 degrees Celsius."
    claims = ext.extract(text)
    assert any(c.claim_type == ClaimType.FACTUAL or c.claim_type == ClaimType.DEFINITIONAL for c in claims)
    print("  Extract factual claims: OK")

def test_extract_numerical_claims():
    ext = ClaimExtractor()
    text = "There are approximately 8 million species on Earth."
    claims = ext.extract(text)
    assert any(c.is_quantified for c in claims)
    print("  Extract numerical claims: OK")

def test_extract_negated_claims():
    ext = ClaimExtractor()
    text = "The Earth is not flat. Python is not a compiled language."
    claims = ext.extract(text)
    assert any(c.is_negated for c in claims)
    print("  Extract negated claims: OK")

def test_extract_entities():
    ext = ClaimExtractor()
    text = "Albert Einstein developed the theory of relativity in 1905."
    claims = ext.extract(text)
    assert any(c.entities for c in claims)
    print("  Extract entities: OK")

def test_extract_opinions():
    ext = ClaimExtractor()
    text = "I think Python is the best programming language."
    claims = ext.extract(text)
    assert any(c.claim_type == ClaimType.OPINION for c in claims)
    print("  Extract opinions: OK")

def test_empty_text():
    ext = ClaimExtractor()
    claims = ext.extract("")
    assert len(claims) == 0
    print("  Empty text: OK")

def test_min_length_filter():
    ext = ClaimExtractor(min_claim_length=20)
    text = "Short."
    claims = ext.extract(text)
    assert len(claims) == 0
    print("  Min length filter: OK")


# ─── ClaimVerifier Tests ────────────────────────────────────────

def test_verifier_basic():
    verifier = ClaimVerifier()
    claim = Claim(
        text="The Earth orbits the Sun.",
        claim_type=ClaimType.FACTUAL,
        severity=ClaimSeverity.MEDIUM,
        confidence=0.8,
        context="The Earth orbits the Sun.",
        position=(0, 25),
    )
    result = verifier.verify(claim)
    assert isinstance(result, VerificationResult)
    assert result.status in (VerificationStatus.SUPPORTED, VerificationStatus.UNVERIFIABLE)
    print("  Verifier basic: OK")

def test_verifier_well_known_fact():
    verifier = ClaimVerifier()
    claim = Claim(
        text="Python is a programming language.",
        claim_type=ClaimType.DEFINITIONAL,
        severity=ClaimSeverity.MEDIUM,
        confidence=0.9,
        context="Python is a programming language.",
        position=(0, 30),
    )
    result = verifier.verify(claim)
    assert result.status == VerificationStatus.SUPPORTED
    print("  Verifier well-known fact: OK")

def test_verifier_batch():
    verifier = ClaimVerifier()
    claims = [
        Claim(text="The Earth is round.", claim_type=ClaimType.FACTUAL,
              severity=ClaimSeverity.MEDIUM, confidence=0.8, context="", position=(0, 18)),
        Claim(text="Python was released in 1991.", claim_type=ClaimType.TEMPORAL,
              severity=ClaimSeverity.MEDIUM, confidence=0.8, context="", position=(0, 28)),
    ]
    results = verifier.verify_batch(claims)
    assert len(results) == 2
    print("  Verifier batch: OK")


# ─── GroundingLayer Tests ───────────────────────────────────────

def test_grounding_basic():
    gl = GroundingLayer()
    result = gl.ground("The Earth is round.")
    assert isinstance(result, GroundingResult)
    assert result.original_text == "The Earth is round."
    assert result.grounded_text is not None
    print("  Grounding basic: OK")

def test_grounding_empty():
    gl = GroundingLayer()
    result = gl.ground("")
    assert result.overall_confidence == 1.0
    print("  Grounding empty: OK")

def test_grounding_score():
    gl = GroundingLayer()
    result = gl.ground("Python is a language.")
    score = result.grounding_score
    assert 0 <= score <= 1
    print("  Grounding score: OK")


# ─── ConfidenceScorer Tests ─────────────────────────────────────

def test_confidence_basic():
    scorer = ConfidenceScorer()
    score = score = scorer.score(
        evidence_count=3,
        source_agreement=0.8,
        claim_text="Python was released in 1991.",
    )
    assert isinstance(score, ConfidenceScore)
    assert 0 <= score.overall <= 1
    print("  Confidence basic: OK")

def test_confidence_high():
    scorer = ConfidenceScorer()
    score = scorer.score(
        evidence_count=5,
        source_agreement=0.95,
        claim_text="The Earth orbits the Sun at approximately 107,000 km/h.",
    )
    assert score.overall > 0.5
    assert score.tier in ("high", "medium")
    print("  Confidence high: OK")

def test_confidence_low():
    scorer = ConfidenceScorer()
    score = scorer.score(
        evidence_count=0,
        source_agreement=0.0,
        claim_text="Maybe aliens exist.",
    )
    assert score.overall < 0.5
    print("  Confidence low: OK")

def test_confidence_signals():
    scorer = ConfidenceScorer()
    score = scorer.score(
        evidence_count=3,
        source_agreement=0.9,
        claim_text="In 2024, the population was 8 billion.",
    )
    assert score.signals is not None
    assert score.explanation
    print("  Confidence signals: OK")

def test_confidence_calibration():
    scorer = ConfidenceScorer()
    for _ in range(10):
        scorer.score(evidence_count=3, source_agreement=0.8, claim_text="test")
    stats = scorer.get_calibration_stats()
    assert stats["count"] == 10
    print("  Confidence calibration: OK")


# ─── RefusalGate Tests ──────────────────────────────────────────

def test_refusal_low_confidence():
    gate = RefusalGate(min_confidence=0.5)
    decision = gate.check("Some claim", confidence=0.3)
    assert decision.should_refuse
    assert decision.reason == RefusalReason.LOW_CONFIDENCE
    print("  Refusal low confidence: OK")

def test_refusal_no_evidence():
    gate = RefusalGate(min_evidence=1)
    decision = gate.check("Some claim", confidence=0.6, evidence_count=0)
    assert decision.should_refuse
    assert decision.reason == RefusalReason.NO_EVIDENCE
    print("  Refusal no evidence: OK")

def test_refusal_safe():
    gate = RefusalGate(min_confidence=0.3)
    decision = gate.check("Some claim", confidence=0.8, evidence_count=3)
    assert not decision.should_refuse
    print("  Refusal safe: OK")

def test_refusal_sensitive_domain():
    gate = RefusalGate(blocked_domains={"medical"})
    decision = gate.check("This medication treats cancer", confidence=0.9, evidence_count=5)
    assert decision.should_refuse
    assert decision.reason == RefusalReason.SENSITIVE_DOMAIN
    print("  Refusal sensitive domain: OK")

def test_refusal_harmful():
    gate = RefusalGate()
    decision = gate.check("How to hack a computer system", confidence=0.9)
    assert decision.should_refuse
    print("  Refusal harmful: OK")

def test_refusal_stats():
    gate = RefusalGate()
    gate.check("test", confidence=0.1)  # Low confidence → refused
    gate.check("test", confidence=0.9, evidence_count=3, verification_status="supported")  # Good → allowed
    stats = gate.get_stats()
    assert stats["total_checks"] == 2
    assert stats["total_refusals"] == 1
    print("  Refusal stats: OK")


# ─── SourceAttribution Tests ────────────────────────────────────

def test_attribution_register():
    sa = SourceAttribution()
    source = Source(id="s1", title="Test Source", url="https://example.com")
    sa.register_source(source)
    assert len(sa.get_all_sources()) == 1
    print("  Attribution register: OK")

def test_attribution_add():
    sa = SourceAttribution()
    source = Source(id="s1", title="Test Source")
    sa.register_source(source)
    attr = sa.add_attribution("Earth orbits Sun", "s1", confidence=0.9)
    assert isinstance(attr, Attribution)
    print("  Attribution add: OK")

def test_attribution_citation():
    sa = SourceAttribution()
    source = Source(id="s1", title="Wikipedia", url="https://en.wikipedia.org")
    sa.register_source(source)
    attr = sa.add_attribution("test", "s1")
    citation = sa.generate_citation(attr, style="inline")
    assert "Wikipedia" in citation
    print("  Attribution citation: OK")

def test_attribution_references():
    sa = SourceAttribution()
    sa.register_source(Source(id="s1", title="Source A"))
    sa.register_source(Source(id="s2", title="Source B"))
    sa.add_attribution("claim1", "s1")
    sa.add_attribution("claim2", "s2")
    refs = sa.generate_references_section()
    assert "Source A" in refs
    print("  Attribution references: OK")

def test_attribution_reliability():
    sa = SourceAttribution()
    sa.register_source(Source(id="s1", title="Test", reliability=0.5))
    sa.update_reliability("s1", was_accurate=True)
    score = sa.get_reliability_score("s1")
    assert score > 0.5
    print("  Attribution reliability: OK")


# ─── AntiHallucinationEngine Tests ──────────────────────────────

def test_engine_init():
    engine = AntiHallucinationEngine()
    assert engine is not None
    print("  Engine init: OK")

def test_engine_verify_output():
    engine = AntiHallucinationEngine()
    report = engine.verify_output("Python is a programming language.")
    assert isinstance(report, VerificationReport)
    assert report.original_text == "Python is a programming language."
    assert report.verified_text is not None
    print("  Engine verify output: OK")

def test_engine_verify_empty():
    engine = AntiHallucinationEngine()
    report = engine.verify_output("")
    assert report.is_safe_to_output
    print("  Engine verify empty: OK")

def test_engine_hallucination_risk():
    engine = AntiHallucinationEngine()
    risk = engine.check_hallucination_risk("Python was created in 1991 by Guido van Rossum.")
    assert "risk_score" in risk
    assert "risk_level" in risk
    print("  Engine hallucination risk: OK")

def test_engine_stats():
    engine = AntiHallucinationEngine()
    engine.verify_output("Test claim.")
    stats = engine.get_stats()
    assert stats["total_verifications"] == 1
    print("  Engine stats: OK")

def test_engine_verify_claims():
    engine = AntiHallucinationEngine()
    results = engine.verify_claims(["The Earth is round.", "Python is a language."])
    assert len(results) == 2
    assert all("status" in r for r in results)
    print("  Engine verify claims: OK")


# ─── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NICTO AI - Anti-Hallucination Engine Test Suite")
    print("=" * 60)

    print("\n--- ClaimExtractor ---")
    test_extract_basic_claims()
    test_extract_factual_claims()
    test_extract_numerical_claims()
    test_extract_negated_claims()
    test_extract_entities()
    test_extract_opinions()
    test_empty_text()
    test_min_length_filter()

    print("\n--- ClaimVerifier ---")
    test_verifier_basic()
    test_verifier_well_known_fact()
    test_verifier_batch()

    print("\n--- GroundingLayer ---")
    test_grounding_basic()
    test_grounding_empty()
    test_grounding_score()

    print("\n--- ConfidenceScorer ---")
    test_confidence_basic()
    test_confidence_high()
    test_confidence_low()
    test_confidence_signals()
    test_confidence_calibration()

    print("\n--- RefusalGate ---")
    test_refusal_low_confidence()
    test_refusal_no_evidence()
    test_refusal_safe()
    test_refusal_sensitive_domain()
    test_refusal_harmful()
    test_refusal_stats()

    print("\n--- SourceAttribution ---")
    test_attribution_register()
    test_attribution_add()
    test_attribution_citation()
    test_attribution_references()
    test_attribution_reliability()

    print("\n--- AntiHallucinationEngine ---")
    test_engine_init()
    test_engine_verify_output()
    test_engine_verify_empty()
    test_engine_hallucination_risk()
    test_engine_stats()
    test_engine_verify_claims()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
