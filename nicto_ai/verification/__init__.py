"""NICTO AI Verification Engine - Anti-hallucination, truth enforcement, fact grounding"""
from .claim_extractor import ClaimExtractor, Claim
from .verifier import ClaimVerifier, VerificationResult
from .grounding import GroundingLayer, GroundingResult
from .confidence import ConfidenceScorer
from .refusal import RefusalGate
from .attribution import SourceAttribution, Source
from .engine import AntiHallucinationEngine

__all__ = [
    "ClaimExtractor",
    "Claim",
    "ClaimVerifier",
    "VerificationResult",
    "GroundingLayer",
    "GroundingResult",
    "ConfidenceScorer",
    "RefusalGate",
    "SourceAttribution",
    "Source",
    "AntiHallucinationEngine",
]
