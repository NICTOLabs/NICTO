"""
NICTO AI - Refusal Gate
Determines when NICTO should refuse to answer or state something.

This is the final safety layer: even if a claim passes verification,
the refusal gate can block it if overall confidence is too low
or if the claim is in a sensitive domain.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RefusalReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    NO_EVIDENCE = "no_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    SAFETY_CRITICAL = "safety_critical"
    SENSITIVE_DOMAIN = "sensitive_domain"
    POTENTIALLY_HARMFUL = "potentially_harmful"
    SPECULATIVE = "speculative"
    OUTDATED = "outdated"


@dataclass
class RefusalDecision:
    """Decision on whether to refuse a claim"""
    should_refuse: bool
    reason: Optional[RefusalReason] = None
    message: str = ""  # What to tell the user
    alternative: str = ""  # What NICTO can do instead
    confidence_in_decision: float = 1.0


class RefusalGate:
    """
    Final safety gate before NICTO outputs a claim.

    The refusal gate checks:
    1. Overall confidence is above minimum threshold
    2. Claim is not in a blocked domain
    3. Claim is not potentially harmful
    4. Claim is not purely speculative
    5. Evidence is not too outdated

    When NICTO refuses, it always explains why and offers alternatives.
    This transparency is critical for trust.
    """

    # Sensitive domains where extra caution is needed
    SENSITIVE_DOMAINS = {
        "medical": ["diagnosis", "treatment", "medication", "dosage", "symptom"],
        "legal": ["legal advice", "lawsuit", "court", "attorney", "rights"],
        "financial": ["investment", "stock", "trading", "financial advice"],
        "safety": ["weapon", "explosive", "dangerous", "toxic", "lethal"],
    }

    # Patterns that indicate potentially harmful content
    HARMFUL_PATTERNS = [
        "how to hack",
        "how to steal",
        "how to harm",
        "how to make a weapon",
        "personal information about",
        "private data of",
    ]

    def __init__(
        self,
        min_confidence: float = 0.4,
        min_evidence: int = 1,
        blocked_domains: Optional[Set[str]] = None,
    ):
        self.min_confidence = min_confidence
        self.min_evidence = min_evidence
        self.blocked_domains = blocked_domains or set()
        self._refusal_count = 0
        self._total_checks = 0

    def check(
        self,
        text: str,
        confidence: float,
        evidence_count: int = 0,
        verification_status: str = "unverifiable",
        claim_type: str = "factual",
    ) -> RefusalDecision:
        """
        Check whether a claim should be refused.

        Args:
            text: The claim text
            confidence: Overall confidence score (0-1)
            evidence_count: Number of evidence pieces
            verification_status: Status from verifier
            claim_type: Type of claim

        Returns:
            RefusalDecision with refuse/allow and explanation
        """
        self._total_checks += 1

        # Check 1: Confidence threshold
        if confidence < self.min_confidence:
            self._refusal_count += 1
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.LOW_CONFIDENCE,
                message=f"Confidence ({confidence:.0%}) is below the minimum threshold ({self.min_confidence:.0%}).",
                alternative="I can share what I know while clearly noting the uncertainty, or I can try to find more information.",
            )

        # Check 2: Evidence threshold
        if evidence_count < self.min_evidence and verification_status == "unverifiable":
            self._refusal_count += 1
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.NO_EVIDENCE,
                message="I cannot verify this claim with available evidence.",
                alternative="I can search for more information or explain what is known about this topic.",
            )

        # Check 3: Conflicting evidence
        if verification_status == "conflicting":
            self._refusal_count += 1
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.CONFLICTING_EVIDENCE,
                message="Available evidence on this topic is conflicting.",
                alternative="I can present different perspectives on this topic, noting where experts disagree.",
            )

        # Check 4: Sensitive domain
        text_lower = text.lower()
        for domain, keywords in self.SENSITIVE_DOMAINS.items():
            if any(kw in text_lower for kw in keywords):
                if domain in self.blocked_domains:
                    self._refusal_count += 1
                    return RefusalDecision(
                        should_refuse=True,
                        reason=RefusalReason.SENSITIVE_DOMAIN,
                        message=f"This falls in the {domain} domain, which requires specialized expertise.",
                        alternative=f"I can provide general information about {domain}, but please consult a qualified professional for specific advice.",
                    )

        # Check 5: Harmful content
        for pattern in self.HARMFUL_PATTERNS:
            if pattern in text_lower:
                self._refusal_count += 1
                return RefusalDecision(
                    should_refuse=True,
                    reason=RefusalReason.POTENTIALLY_HARMFUL,
                    message="This request may involve potentially harmful content.",
                    alternative="I can help with constructive and safe applications of this knowledge.",
                )

        # Check 6: Purely speculative
        speculative_words = ["might be", "could be", "maybe", "just guessing", "no idea but"]
        if any(w in text_lower for w in speculative_words):
            self._refusal_count += 1
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.SPECULATIVE,
                message="This appears to be speculative rather than evidence-based.",
                alternative="I can share what is known, or I can explain the reasoning behind different possibilities.",
            )

        # Allow the claim
        return RefusalDecision(
            should_refuse=False,
            confidence_in_decision=confidence,
        )

    def get_stats(self) -> Dict:
        """Get refusal statistics."""
        return {
            "total_checks": self._total_checks,
            "total_refusals": self._refusal_count,
            "refusal_rate": self._refusal_count / max(1, self._total_checks),
        }
