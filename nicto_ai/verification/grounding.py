"""
NICTO AI - Grounding Layer
Ensures all NICTO outputs are grounded in verified evidence.

The grounding layer sits between NICTO's generation and output,
intercepting ungrounded claims and forcing them through verification.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .claim_extractor import ClaimExtractor, Claim, ClaimSeverity
from .verifier import ClaimVerifier, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of grounding a piece of text"""
    original_text: str
    grounded_text: str  # Text with claims verified/qualified/removed
    claims: List[Claim] = field(default_factory=list)
    verifications: List[VerificationResult] = field(default_factory=list)
    overall_confidence: float = 0.0
    n_safe: int = 0  # Claims safe to state
    n_qualified: int = 0  # Claims that needed qualification
    n_refused: int = 0  # Claims that were refused
    n_unverifiable: int = 0  # Claims that couldn't be verified

    @property
    def grounding_score(self) -> float:
        """Score from 0-1 indicating how well-grounded the text is."""
        if not self.claims:
            return 1.0  # No claims = fully grounded (nothing to verify)
        total = len(self.claims)
        return (self.n_safe + self.n_qualified * 0.5) / total


class GroundingLayer:
    """
    Grounds NICTO's output in verified evidence.

    This is the core anti-hallucination mechanism. It:
    1. Extracts claims from generated text
    2. Verifies each claim against evidence
    3. Rewrites text to qualify or remove ungrounded claims
    4. Returns a confidence-scored, grounded version

    The grounding layer NEVER removes information arbitrarily.
    It either verifies, qualifies, or refuses - transparently.
    """

    def __init__(
        self,
        claim_extractor: Optional[ClaimExtractor] = None,
        claim_verifier: Optional[ClaimVerifier] = None,
        grounding_threshold: float = 0.5,
        refuse_threshold: float = 0.3,
    ):
        """
        Args:
            claim_extractor: ClaimExtractor instance (created if None)
            claim_verifier: ClaimVerifier instance (created if None)
            grounding_threshold: Minimum confidence to ground a claim
            refuse_threshold: Below this confidence, refuse the claim
        """
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.claim_verifier = claim_verifier
        self.grounding_threshold = grounding_threshold
        self.refuse_threshold = refuse_threshold

    def ground(self, text: str) -> GroundingResult:
        """
        Ground a piece of text by verifying all claims.

        Args:
            text: Generated text to ground

        Returns:
            GroundingResult with grounded text and verification details
        """
        if not text or not text.strip():
            return GroundingResult(
                original_text=text,
                grounded_text=text,
                overall_confidence=1.0,
            )

        # Step 1: Extract claims
        claims = self.claim_extractor.extract(text)

        if not claims:
            return GroundingResult(
                original_text=text,
                grounded_text=text,
                claims=[],
                overall_confidence=1.0,
                n_safe=0,
                n_qualified=0,
                n_refused=0,
                n_unverifiable=0,
            )

        # Step 2: Verify each claim
        verifications = []
        if self.claim_verifier is not None:
            verifications = self.claim_verifier.verify_batch(claims)
        else:
            # Without a verifier, mark all as unverifiable
            verifications = [
                VerificationResult(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIABLE,
                    confidence=0.0,
                    explanation="No verifier available.",
                    recommendation="QUALIFY: Cannot verify without evidence sources.",
                )
                for claim in claims
            ]

        # Step 3: Classify each claim
        n_safe = 0
        n_qualified = 0
        n_refused = 0
        n_unverifiable = 0

        for v in verifications:
            if v.status == VerificationStatus.SUPPORTED and v.confidence > self.grounding_threshold:
                n_safe += 1
            elif v.status == VerificationStatus.REFUTED or v.confidence < self.refuse_threshold:
                n_refused += 1
            elif v.status == VerificationStatus.UNVERIFIABLE:
                n_unverifiable += 1
            else:
                n_qualified += 1

        # Step 4: Rewrite text with qualifications
        grounded_text = self._rewrite_with_qualifications(text, claims, verifications)

        # Step 5: Compute overall confidence
        if verifications:
            overall_confidence = sum(v.confidence for v in verifications) / len(verifications)
        else:
            overall_confidence = 1.0

        return GroundingResult(
            original_text=text,
            grounded_text=grounded_text,
            claims=claims,
            verifications=verifications,
            overall_confidence=overall_confidence,
            n_safe=n_safe,
            n_qualified=n_qualified,
            n_refused=n_refused,
            n_unverifiable=n_unverifiable,
        )

    def _rewrite_with_qualifications(
        self, text: str, claims: List[Claim],
        verifications: List[VerificationResult],
    ) -> str:
        """Rewrite text, adding qualifiers to unverified claims."""
        if not verifications:
            return text

        result = text

        # Process in reverse order to preserve positions
        sorted_pairs = sorted(
            zip(claims, verifications),
            key=lambda p: p[0].position[0],
            reverse=True,
        )

        for claim, verification in sorted_pairs:
            if verification.status == VerificationStatus.SUPPORTED and verification.confidence > 0.8:
                # Safe to state as-is
                continue

            if verification.status == VerificationStatus.REFUTED or verification.confidence < self.refuse_threshold:
                # Replace with refusal
                replacement = self._generate_refusal(claim)
                start, end = claim.position
                if start >= 0 and end <= len(result):
                    result = result[:start] + replacement + result[end:]

            elif verification.status == VerificationStatus.CONFLICTING:
                # Add qualification
                qualification = self._generate_qualification(claim, "conflicting evidence exists")
                start, end = claim.position
                if start >= 0 and end <= len(result):
                    # Insert qualification before the claim
                    period_pos = result.rfind(".", 0, start)
                    if period_pos >= 0:
                        result = result[:period_pos + 1] + " " + qualification + " " + result[period_pos + 1:]

            elif verification.status == VerificationStatus.UNVERIFIABLE:
                # Add hedging
                qualification = self._generate_qualification(claim, "this is uncertain")
                start, end = claim.position
                if start >= 0 and end <= len(result):
                    period_pos = result.rfind(".", 0, start)
                    if period_pos >= 0:
                        result = result[:period_pos + 1] + " " + qualification + " " + result[period_pos + 1:]

        return result

    def _generate_refusal(self, claim: Claim) -> str:
        """Generate a refusal statement for a claim that can't be verified."""
        if claim.claim_type.value in ("statistical", "numerical"):
            return "[This statistical claim could not be verified and has been omitted]"
        elif claim.claim_type.value == "temporal":
            return "[This historical claim could not be verified and has been omitted]"
        elif claim.claim_type.value == "causal":
            return "[This causal claim could not be verified and has been omitted]"
        else:
            return "[This claim could not be verified and has been omitted]"

    def _generate_qualification(self, claim: Claim, reason: str) -> str:
        """Generate a qualification for a claim that needs hedging."""
        qualifiers = [
            f"Note: {reason}.",
            f"According to available evidence ({reason}):",
            f"It appears that (though {reason}):",
        ]
        # Pick based on claim type
        if claim.claim_type.value == "statistical":
            return f"[Note: {reason}. These figures should be verified.]"
        elif claim.claim_type.value == "causal":
            return f"[Note: {reason}. The causal relationship is not fully established.]"
        else:
            return f"[Note: {reason}.]"
