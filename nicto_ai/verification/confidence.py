"""
NICTO AI - Confidence Scorer
Computes multi-signal confidence scores for claims and outputs.

Combines multiple independent signals to produce a reliable
confidence estimate that reflects actual reliability.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceSignals:
    """Individual confidence signals"""
    evidence_strength: float = 0.0    # How strong is the evidence
    source_agreement: float = 0.0     # Do sources agree
    claim_specificity: float = 0.0    # How specific/verifiable is the claim
    temporal_relevance: float = 0.0   # How current is the information
    internal_consistency: float = 0.0 # Is it consistent with known facts
    linguistic_certainty: float = 0.0 # Does the language express certainty


@dataclass
class ConfidenceScore:
    """Final confidence score with breakdown"""
    overall: float  # 0-1 final confidence
    signals: ConfidenceSignals = field(default_factory=ConfidenceSignals)
    tier: str = "low"  # "high", "medium", "low", "very_low"
    explanation: str = ""

    @property
    def is_high_confidence(self) -> bool:
        return self.overall > 0.8

    @property
    def is_actionable(self) -> bool:
        return self.overall > 0.5


class ConfidenceScorer:
    """
    Multi-signal confidence scoring for NICTO's claims.

    Instead of a single learned confidence value (which can be miscalibrated),
    this scorer combines multiple independent signals:

    1. Evidence strength: How many sources support the claim
    2. Source agreement: Do different sources agree
    3. Claim specificity: Specific claims are easier to verify
    4. Temporal relevance: Recent information is more reliable
    5. Internal consistency: Does it match known facts
    6. Linguistic certainty: Does the source language express confidence

    The combination is intentionally conservative - it's better to
    be under-confident than over-confident.
    """

    # Weights for combining signals (tuned for conservatism)
    SIGNAL_WEIGHTS = {
        "evidence_strength": 0.25,
        "source_agreement": 0.20,
        "claim_specificity": 0.15,
        "temporal_relevance": 0.15,
        "internal_consistency": 0.15,
        "linguistic_certainty": 0.10,
    }

    def __init__(self, conservative: bool = True):
        """
        Args:
            conservative: If True, bias toward lower confidence (safer)
        """
        self.conservative = conservative
        self._history: List[ConfidenceScore] = []

    def score(
        self,
        evidence_count: int = 0,
        source_agreement: float = 0.5,
        claim_text: str = "",
        evidence_texts: Optional[List[str]] = None,
        known_facts: Optional[Dict[str, bool]] = None,
    ) -> ConfidenceScore:
        """
        Compute a multi-signal confidence score.

        Args:
            evidence_count: Number of evidence pieces found
            source_agreement: 0-1, how much sources agree (1=full agreement)
            claim_text: The claim text for linguistic analysis
            evidence_texts: List of evidence texts for cross-reference
            known_facts: Dict of known facts for consistency checking

        Returns:
            ConfidenceScore with overall score and signal breakdown
        """
        signals = ConfidenceSignals()

        # Signal 1: Evidence strength (more evidence = higher confidence)
        signals.evidence_strength = self._compute_evidence_strength(evidence_count)

        # Signal 2: Source agreement
        signals.source_agreement = source_agreement

        # Signal 3: Claim specificity (specific = easier to verify)
        signals.claim_specificity = self._compute_specificity(claim_text)

        # Signal 4: Temporal relevance
        signals.temporal_relevance = self._compute_temporal_relevance(claim_text, evidence_texts)

        # Signal 5: Internal consistency
        signals.internal_consistency = self._compute_consistency(claim_text, known_facts)

        # Signal 6: Linguistic certainty
        signals.linguistic_certainty = self._compute_linguistic_certainty(claim_text)

        # Combine signals
        overall = self._combine_signals(signals)

        # Apply conservatism bias
        if self.conservative:
            overall = overall * 0.9  # 10% conservative bias

        # Determine tier
        tier = self._determine_tier(overall)

        # Generate explanation
        explanation = self._generate_explanation(signals, overall, tier)

        score = ConfidenceScore(
            overall=overall,
            signals=signals,
            tier=tier,
            explanation=explanation,
        )

        self._history.append(score)
        return score

    def _compute_evidence_strength(self, evidence_count: int) -> float:
        """Map evidence count to a 0-1 strength score."""
        # Diminishing returns: 1 source = 0.4, 2 = 0.6, 3 = 0.75, 5+ = 0.9
        if evidence_count == 0:
            return 0.0
        return min(0.9, 0.4 + 0.2 * math.log2(evidence_count))

    def _compute_specificity(self, claim_text: str) -> float:
        """How specific and verifiable is the claim."""
        specificity = 0.5  # baseline

        # Numbers make claims more specific
        import re
        if re.search(r"\d+", claim_text):
            specificity += 0.2

        # Named entities make claims more specific
        if re.search(r"\b[A-Z][a-z]+\b", claim_text):
            specificity += 0.1

        # Dates make claims more specific
        if re.search(r"\b(?:19|20)\d{2}\b", claim_text):
            specificity += 0.1

        # Vague language reduces specificity
        vague_words = ["maybe", "perhaps", "might", "could", "some", "many", "often"]
        if any(w in claim_text.lower() for w in vague_words):
            specificity -= 0.2

        return max(0.0, min(1.0, specificity))

    def _compute_temporal_relevance(self, claim_text: str, evidence_texts: Optional[List[str]]) -> float:
        """How current is the information."""
        import re

        # Check for year references in claim
        year_match = re.search(r"\b(19|20)(\d{2})\b", claim_text)
        if year_match:
            year = int(year_match.group(0))
            # Assume current year is ~2026
            age = 2026 - year
            if age < 0:
                return 0.3  # Future claim
            elif age < 2:
                return 0.9
            elif age < 5:
                return 0.7
            elif age < 10:
                return 0.5
            else:
                return 0.3

        # Check for time-sensitive keywords
        temporal_words = ["today", "now", "currently", "recent", "latest", "new"]
        if any(w in claim_text.lower() for w in temporal_words):
            return 0.8  # Claims about current state need recent evidence

        return 0.6  # Default: moderately relevant

    def _compute_consistency(self, claim_text: str, known_facts: Optional[Dict[str, bool]]) -> float:
        """How consistent is the claim with known facts."""
        if not known_facts:
            return 0.5  # No information

        # Simple keyword matching against known facts
        claim_lower = claim_text.lower()
        consistent = 0
        total = 0

        for fact, is_true in known_facts.items():
            if fact.lower() in claim_lower:
                total += 1
                if is_true:
                    consistent += 1

        if total == 0:
            return 0.5
        return consistent / total

    def _compute_linguistic_certainty(self, claim_text: str) -> float:
        """How certain does the language sound."""
        certain_words = ["is", "are", "was", "were", "has", "have", "will", "definitely", "certainly"]
        uncertain_words = ["might", "could", "may", "possibly", "perhaps", "seems", "appears", "likely"]

        claim_lower = claim_text.lower()
        certain_count = sum(1 for w in certain_words if w in claim_lower)
        uncertain_count = sum(1 for w in uncertain_words if w in claim_lower)

        total = certain_count + uncertain_count
        if total == 0:
            return 0.5

        return certain_count / total

    def _combine_signals(self, signals: ConfidenceSignals) -> float:
        """Combine individual signals into overall confidence."""
        weighted_sum = (
            self.SIGNAL_WEIGHTS["evidence_strength"] * signals.evidence_strength +
            self.SIGNAL_WEIGHTS["source_agreement"] * signals.source_agreement +
            self.SIGNAL_WEIGHTS["claim_specificity"] * signals.claim_specificity +
            self.SIGNAL_WEIGHTS["temporal_relevance"] * signals.temporal_relevance +
            self.SIGNAL_WEIGHTS["internal_consistency"] * signals.internal_consistency +
            self.SIGNAL_WEIGHTS["linguistic_certainty"] * signals.linguistic_certainty
        )

        # Penalize if any critical signal is very low
        min_signal = min(
            signals.evidence_strength,
            signals.source_agreement,
            signals.internal_consistency,
        )
        if min_signal < 0.2:
            weighted_sum *= 0.7  # Penalty for critical weakness

        return max(0.0, min(1.0, weighted_sum))

    def _determine_tier(self, score: float) -> str:
        """Map score to a confidence tier."""
        if score > 0.8:
            return "high"
        elif score > 0.6:
            return "medium"
        elif score > 0.4:
            return "low"
        else:
            return "very_low"

    def _generate_explanation(self, signals: ConfidenceSignals, overall: float, tier: str) -> str:
        """Generate human-readable explanation."""
        strengths = []
        weaknesses = []

        if signals.evidence_strength > 0.7:
            strengths.append("strong evidence")
        elif signals.evidence_strength < 0.3:
            weaknesses.append("weak evidence")

        if signals.source_agreement > 0.8:
            strengths.append("sources agree")
        elif signals.source_agreement < 0.4:
            weaknesses.append("sources disagree")

        if signals.internal_consistency > 0.7:
            strengths.append("consistent with known facts")
        elif signals.internal_consistency < 0.3:
            weaknesses.append("inconsistent with known facts")

        parts = [f"Overall confidence: {overall:.0%} ({tier})"]
        if strengths:
            parts.append(f"Strengths: {', '.join(strengths)}")
        if weaknesses:
            parts.append(f"Weaknesses: {', '.join(weaknesses)}")

        return ". ".join(parts)

    def get_calibration_stats(self) -> Dict:
        """Get calibration statistics from history."""
        if not self._history:
            return {"count": 0}

        scores = [s.overall for s in self._history]
        return {
            "count": len(scores),
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "high_confidence_pct": sum(1 for s in scores if s > 0.8) / len(scores),
        }
