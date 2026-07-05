"""
NICTO AI - Anti-Hallucination Engine
The master orchestrator that ensures NICTO never hallucinates, never lies,
and never builds fake things.

This engine wraps NICTO's output pipeline and enforces:
1. Every claim must be extractable and verifiable
2. Every output must be grounded in evidence
3. Every source must be attributed
4. Uncertainty must be expressed honestly
5. Refusals must be transparent

This is the core of NICTO's truthfulness system.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .claim_extractor import ClaimExtractor, Claim, ClaimSeverity
from .verifier import ClaimVerifier, VerificationResult, VerificationStatus
from .grounding import GroundingLayer, GroundingResult
from .confidence import ConfidenceScorer, ConfidenceScore
from .refusal import RefusalGate, RefusalDecision
from .attribution import SourceAttribution, Source

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    """Complete verification report for a piece of text"""
    original_text: str
    verified_text: str  # Text after verification (may be modified)
    grounding_result: Optional[GroundingResult] = None
    refusal_decision: Optional[RefusalDecision] = None
    overall_confidence: float = 0.0
    is_safe_to_output: bool = True
    processing_time_ms: float = 0.0
    claims_found: int = 0
    claims_verified: int = 0
    claims_refused: int = 0
    sources_used: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"Verification: {self.claims_found} claims found, "
            f"{self.claims_verified} verified, {self.claims_refused} refused. "
            f"Confidence: {self.overall_confidence:.0%}. "
            f"Safe to output: {self.is_safe_to_output}"
        )


class AntiHallucinationEngine:
    """
    NICTO's Anti-Hallucination Engine - the truth enforcement system.

    This engine is the final gatekeeper between NICTO's internal processing
    and what gets presented to users. It ensures:

    1. NO unverified claims leave NICTO
    2. ALL uncertainty is expressed honestly
    3. ALL sources are attributed
    4. REFUSALS are transparent and helpful
    5. CONFIDENCE scores reflect actual reliability

    Architecture:
    ┌─────────────────┐
    │  NICTO Output    │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Claim Extraction │ ← Extract atomic claims
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Claim Verification│ ← Verify against evidence
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Confidence Scoring│ ← Multi-signal confidence
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Refusal Gate     │ ← Block unverifiable claims
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Grounding Layer  │ ← Rewrite with qualifications
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Source Attribution│ ← Cite all sources
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  Verified Output │
    └─────────────────┘
    """

    def __init__(
        self,
        knowledge_base=None,
        browser=None,
        min_confidence: float = 0.4,
        grounding_threshold: float = 0.5,
        refuse_threshold: float = 0.3,
        enable_web_verification: bool = True,
    ):
        """
        Args:
            knowledge_base: NICTOKnowledgeBase for local knowledge
            browser: NICTOBrowser for web search verification
            min_confidence: Minimum confidence to allow a claim
            grounding_threshold: Threshold for grounding claims
            refuse_threshold: Below this, refuse the claim
            enable_web_verification: Whether to use web search
        """
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier(
            knowledge_base=knowledge_base,
            browser=browser if enable_web_verification else None,
        )
        self.grounding_layer = GroundingLayer(
            claim_extractor=self.claim_extractor,
            claim_verifier=self.claim_verifier,
            grounding_threshold=grounding_threshold,
            refuse_threshold=refuse_threshold,
        )
        self.confidence_scorer = ConfidenceScorer(conservative=True)
        self.refusal_gate = RefusalGate(
            min_confidence=min_confidence,
        )
        self.source_attribution = SourceAttribution()

        self._total_verifications = 0
        self._total_refusals = 0
        self._total_groundings = 0

    def verify_output(self, text: str) -> VerificationReport:
        """
        Verify a complete piece of NICTO output.

        This is the main entry point. Call this on every output
        before presenting it to users.

        Args:
            text: The text output to verify

        Returns:
            VerificationReport with verification details
        """
        start_time = time.time()
        report = VerificationReport(original_text=text, verified_text=text)

        if not text or not text.strip():
            report.verified_text = text
            report.is_safe_to_output = True
            return report

        # Step 1: Extract claims
        claims = self.claim_extractor.extract(text)
        report.claims_found = len(claims)

        # Step 2: Ground the text (verify + rewrite)
        grounding_result = self.grounding_layer.ground(text)
        report.grounding_result = grounding_result
        report.verified_text = grounding_result.grounded_text
        report.overall_confidence = grounding_result.overall_confidence
        report.claims_verified = grounding_result.n_safe + grounding_result.n_qualified
        report.claims_refused = grounding_result.n_refused

        # Step 3: Check refusal gate
        if grounding_result.n_refused > 0 or grounding_result.overall_confidence < 0.3:
            refusal = self.refusal_gate.check(
                text=text,
                confidence=grounding_result.overall_confidence,
                evidence_count=len(grounding_result.verifications),
                verification_status=self._get_overall_status(grounding_result.verifications),
            )
            report.refusal_decision = refusal

            if refusal.should_refuse:
                report.is_safe_to_output = False
                report.warnings.append(f"Refused: {refusal.message}")
                self._total_refusals += 1

        # Step 4: Add source attributions
        for verification in grounding_result.verifications:
            for source_str in verification.sources:
                source = Source(
                    id=source_str,
                    title=source_str,
                    source_type="web" if source_str.startswith("web:") else "knowledge_base",
                )
                self.source_attribution.register_source(source)
                self.source_attribution.add_attribution(
                    claim_text=verification.claim.text,
                    source_id=source.id,
                    confidence=verification.confidence,
                )

        report.sources_used = len(self.source_attribution.get_all_sources())

        # Step 5: Final safety check
        if report.overall_confidence < 0.2 and report.claims_found > 0:
            report.is_safe_to_output = False
            report.warnings.append("Overall confidence too low for output")

        # Step 6: Add citations to verified text
        if report.is_safe_to_output and report.sources_used > 0:
            citations = self.source_attribution.generate_references_section(style="numbered")
            if citations:
                report.verified_text = report.verified_text + "\n\n" + citations

        # Timing
        report.processing_time_ms = (time.time() - start_time) * 1000

        # Update stats
        self._total_verifications += 1
        self._total_groundings += 1

        logger.info(
            "Verification complete: %d claims, %.0f%% confidence, safe=%s, time=%.1fms",
            report.claims_found,
            report.overall_confidence * 100,
            report.is_safe_to_output,
            report.processing_time_ms,
        )

        return report

    def verify_claims(self, claims: List[str]) -> List[Dict]:
        """Verify a list of individual claims."""
        results = []
        for claim_text in claims:
            claim = Claim(
                text=claim_text,
                claim_type=None,  # Will be auto-detected
                severity=ClaimSeverity.MEDIUM,
                confidence=0.5,
                context=claim_text,
                position=(0, len(claim_text)),
            )
            verification = self.claim_verifier.verify(claim)
            confidence = self.confidence_scorer.score(
                evidence_count=len(verification.evidence),
                source_agreement=verification.confidence,
                claim_text=claim_text,
            )
            refusal = self.refusal_gate.check(
                text=claim_text,
                confidence=confidence.overall,
                evidence_count=len(verification.evidence),
                verification_status=verification.status.value,
            )
            results.append({
                "claim": claim_text,
                "status": verification.status.value,
                "confidence": confidence.overall,
                "confidence_tier": confidence.tier,
                "should_refuse": refusal.should_refuse,
                "refusal_reason": refusal.reason.value if refusal.reason else None,
                "explanation": verification.explanation,
            })
        return results

    def check_hallucination_risk(self, text: str) -> Dict:
        """
        Quick check for hallucination risk without full verification.
        Useful for real-time monitoring during generation.
        """
        claims = self.claim_extractor.extract(text)

        risk_score = 0.0
        risk_factors = []

        # Factor 1: Number of claims (more claims = more risk)
        if len(claims) > 5:
            risk_score += 0.2
            risk_factors.append(f"High claim count: {len(claims)}")

        # Factor 2: Specificity of claims
        specific_claims = [c for c in claims if c.is_quantified or c.entities]
        if len(specific_claims) > 3:
            risk_score += 0.2
            risk_factors.append(f"Many specific claims: {len(specific_claims)}")

        # Factor 3: Severity distribution
        high_severity = [c for c in claims if c.severity in (ClaimSeverity.HIGH, ClaimSeverity.SAFETY)]
        if high_severity:
            risk_score += 0.3
            risk_factors.append(f"High-severity claims: {len(high_severity)}")

        # Factor 4: Negated claims (often wrong)
        negated = [c for c in claims if c.is_negated]
        if negated:
            risk_score += 0.1
            risk_factors.append(f"Negated claims: {len(negated)}")

        # Factor 5: Causal claims (often oversimplified)
        causal = [c for c in claims if c.claim_type and c.claim_type.value == "causal"]
        if causal:
            risk_score += 0.15
            risk_factors.append(f"Causal claims: {len(causal)}")

        return {
            "risk_score": min(1.0, risk_score),
            "risk_level": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low",
            "risk_factors": risk_factors,
            "claims_found": len(claims),
        }

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "total_verifications": self._total_verifications,
            "total_refusals": self._total_refusals,
            "total_groundings": self._total_groundings,
            "refusal_rate": self._total_refusals / max(1, self._total_verifications),
            "attribution_stats": self.source_attribution.get_attribution_stats(),
            "confidence_stats": self.confidence_scorer.get_calibration_stats(),
            "refusal_stats": self.refusal_gate.get_stats(),
        }

    def _get_overall_status(self, verifications: List[VerificationResult]) -> str:
        """Get the overall verification status from a list of results."""
        if not verifications:
            return "unverifiable"

        statuses = [v.status for v in verifications]
        if all(s == VerificationStatus.SUPPORTED for s in statuses):
            return "supported"
        elif any(s == VerificationStatus.REFUTED for s in statuses):
            return "refuted"
        elif any(s == VerificationStatus.CONFLICTING for s in statuses):
            return "conflicting"
        else:
            return "unverifiable"
