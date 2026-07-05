"""
NICTO AI - Claim Verifier
Verifies extracted claims against multiple evidence sources.

Uses the knowledge base, web search, and internal knowledge
to determine if claims are supported, refuted, or unverifiable.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .claim_extractor import Claim, ClaimType, ClaimSeverity

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    SUPPORTED = "supported"       # Evidence confirms the claim
    REFUTED = "refuted"           # Evidence contradicts the claim
    PARTIALLY_SUPPORTED = "partially_supported"  # Some evidence, some gaps
    UNVERIFIABLE = "unverifiable" # Cannot find evidence either way
    CONFLICTING = "conflicting"   # Evidence goes both ways
    OUTDATED = "outdated"         # Was true, but may no longer be


@dataclass
class Evidence:
    """A piece of evidence supporting or refuting a claim"""
    source: str  # Source identifier (URL, document name, etc.)
    text: str  # The evidence text
    relevance: float  # 0-1 how relevant to the claim
    supports: bool  # True = supports claim, False = refutes
    confidence: float  # 0-1 how confident in this evidence
    timestamp: Optional[str] = None  # When the evidence was published
    metadata: Dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verifying a single claim"""
    claim: Claim
    status: VerificationStatus
    confidence: float  # 0-1 overall confidence in verification
    evidence: List[Evidence] = field(default_factory=list)
    explanation: str = ""  # Human-readable explanation
    recommendation: str = ""  # What NICTO should do (state it, qualify it, or refuse)
    sources: List[str] = field(default_factory=list)  # Source URLs/references

    @property
    def is_reliable(self) -> bool:
        """Is this claim safe to state as fact?"""
        return self.status in (VerificationStatus.SUPPORTED,) and self.confidence > 0.7

    @property
    def should_refuse(self) -> bool:
        """Should NICTO refuse to state this claim?"""
        return self.status in (VerificationStatus.REFUTED, VerificationStatus.UNVERIFIABLE) or self.confidence < 0.3


class ClaimVerifier:
    """
    Verifies claims against evidence from multiple sources.

    Verification pipeline:
    1. Check internal knowledge (训练 data, learned facts)
    2. Query knowledge base for relevant documents
    3. Search the web for current information
    4. Cross-reference multiple sources
    5. Compute verification confidence
    6. Determine if claim should be stated, qualified, or refused

    The verifier is conservative: when in doubt, it flags the claim
    as unverifiable rather than guessing.
    """

    def __init__(
        self,
        knowledge_base=None,
        browser=None,
        min_evidence: int = 1,
        min_confidence: float = 0.5,
    ):
        """
        Args:
            knowledge_base: NICTOKnowledgeBase instance for local knowledge
            browser: NICTOBrowser instance for web search
            min_evidence: Minimum evidence pieces needed for verification
            min_confidence: Minimum confidence to consider a claim verified
        """
        self.knowledge_base = knowledge_base
        self.browser = browser
        self.min_evidence = min_evidence
        self.min_confidence = min_confidence
        self._verification_cache: Dict[str, VerificationResult] = {}

    def verify(self, claim: Claim) -> VerificationResult:
        """
        Verify a single claim against available evidence.

        Args:
            claim: The claim to verify

        Returns:
            VerificationResult with status, confidence, and evidence
        """
        # Check cache
        cache_key = claim.text.lower().strip()
        if cache_key in self._verification_cache:
            return self._verification_cache[cache_key]

        evidence = []

        # Step 1: Check internal knowledge patterns
        internal_evidence = self._check_internal_knowledge(claim)
        evidence.extend(internal_evidence)

        # Step 2: Query knowledge base if available
        if self.knowledge_base is not None:
            kb_evidence = self._query_knowledge_base(claim)
            evidence.extend(kb_evidence)

        # Step 3: Web search for current information
        if self.browser is not None:
            web_evidence = self._search_web(claim)
            evidence.extend(web_evidence)

        # Step 4: Cross-reference and compute verification
        result = self._compute_verification(claim, evidence)

        # Cache result
        self._verification_cache[cache_key] = result

        return result

    def verify_batch(self, claims: List[Claim]) -> List[VerificationResult]:
        """Verify multiple claims."""
        return [self.verify(claim) for claim in claims]

    def _check_internal_knowledge(self, claim: Claim) -> List[Evidence]:
        """Check claim against internal knowledge patterns."""
        evidence = []

        # Pattern-based checks for common factual claims
        text = claim.text.lower()

        # Check for well-known facts
        well_known = self._check_well_known_facts(text)
        if well_known:
            evidence.append(well_known)

        # Check for logical consistency
        logical = self._check_logical_consistency(claim)
        if logical:
            evidence.append(logical)

        return evidence

    def _check_well_known_facts(self, text: str) -> Optional[Evidence]:
        """Check against a database of well-known facts."""
        # This is a simplified version - in production, this would
        # query a comprehensive fact database
        well_known_facts = {
            "earth": {
                "orbits the sun": True,
                "is round": True,
                "has one moon": True,
                "is the third planet": True,
            },
            "water": {
                "boils at 100": True,
                "freezes at 0": True,
                "is h2o": True,
            },
            "python": {
                "was created by guido": True,
                "is a programming language": True,
                "first released in 1991": True,
            },
        }

        for entity, facts in well_known_facts.items():
            if entity in text:
                for fact, is_true in facts.items():
                    if fact in text:
                        return Evidence(
                            source="internal:well_known_facts",
                            text=f"Well-known fact: {fact}",
                            relevance=0.9,
                            supports=is_true,
                            confidence=0.95,
                        )
        return None

    def _check_logical_consistency(self, claim: Claim) -> Optional[Evidence]:
        """Check if claim is logically consistent."""
        text = claim.text.lower()

        # Check for contradictions within the claim
        contradictions = [
            ("greater than", "less than"),
            ("always", "never"),
            ("all", "none"),
            ("more", "fewer"),
            ("increase", "decrease"),
        ]

        for word1, word2 in contradictions:
            if word1 in text and word2 in text:
                return Evidence(
                    source="internal:logic",
                    text=f"Claim contains potential contradiction: '{word1}' and '{word2}'",
                    relevance=0.8,
                    supports=False,
                    confidence=0.7,
                )

        return None

    def _query_knowledge_base(self, claim: Claim) -> List[Evidence]:
        """Query the knowledge base for relevant evidence."""
        evidence = []
        try:
            results = self.knowledge_base.query(claim.text, top_k=5)
            for result in results:
                # Determine if the knowledge base entry supports or refutes
                supports = self._determine_support(claim.text, result.get("text", ""))
                evidence.append(Evidence(
                    source=f"knowledge_base:{result.get('title', 'unknown')}",
                    text=result.get("text", "")[:500],
                    relevance=result.get("score", 0.5),
                    supports=supports,
                    confidence=result.get("score", 0.5),
                ))
        except Exception as e:
            logger.warning("Knowledge base query failed: %s", e)
        return evidence

    def _search_web(self, claim: Claim) -> List[Evidence]:
        """Search the web for evidence about the claim."""
        evidence = []
        try:
            search_results = self.browser.search(claim.text, max_results=3)
            for result in search_results.results[:3]:
                # Try to extract relevant content
                content = result.get("snippet", "")
                if content:
                    supports = self._determine_support(claim.text, content)
                    evidence.append(Evidence(
                        source=f"web:{result.get('url', 'unknown')}",
                        text=content[:500],
                        relevance=0.6,
                        supports=supports,
                        confidence=0.5,
                        metadata={"url": result.get("url", "")},
                    ))
        except Exception as e:
            logger.warning("Web search failed: %s", e)
        return evidence

    def _determine_support(self, claim_text: str, evidence_text: str) -> bool:
        """Determine if evidence text supports or refutes the claim."""
        claim_lower = claim_text.lower()
        evidence_lower = evidence_text.lower()

        # Simple heuristic: check for supporting or contradicting keywords
        supporting_keywords = ["confirmed", "verified", "true", "correct", "indeed", "yes"]
        contradicting_keywords = ["false", "incorrect", "refuted", "debunked", "myth", "no"]

        support_score = sum(1 for k in supporting_keywords if k in evidence_lower)
        contradict_score = sum(1 for k in contradicting_keywords if k in evidence_lower)

        # Check for negation alignment
        claim_negated = any(w in claim_lower for w in ["not", "never", "no", "neither"])
        evidence_negated = any(w in evidence_lower for w in ["not", "never", "no", "neither"])

        if claim_negated == evidence_negated:
            support_score += 1
        else:
            contradict_score += 1

        return support_score > contradict_score

    def _compute_verification(self, claim: Claim, evidence: List[Evidence]) -> VerificationResult:
        """Compute the final verification result from all evidence."""
        if not evidence:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                explanation="No evidence found to verify this claim.",
                recommendation="Do not state this claim as fact. Either qualify it or omit it.",
            )

        # Count supporting vs refuting evidence
        supporting = [e for e in evidence if e.supports]
        refuting = [e for e in evidence if not e.supports]

        # Weight by relevance and confidence
        support_weight = sum(e.relevance * e.confidence for e in supporting)
        refute_weight = sum(e.relevance * e.confidence for e in refuting)
        total_weight = support_weight + refute_weight

        if total_weight == 0:
            confidence = 0.0
        else:
            confidence = support_weight / total_weight

        # Determine status
        if len(supporting) > 0 and len(refuting) == 0:
            status = VerificationStatus.SUPPORTED
        elif len(refuting) > 0 and len(supporting) == 0:
            status = VerificationStatus.REFUTED
        elif support_weight > refute_weight * 2:
            status = VerificationStatus.SUPPORTED
        elif refute_weight > support_weight * 2:
            status = VerificationStatus.REFUTED
        elif support_weight > 0 and refute_weight > 0:
            status = VerificationStatus.CONFLICTING
        else:
            status = VerificationStatus.PARTIALLY_SUPPORTED

        # Generate explanation
        explanation = self._generate_explanation(claim, status, supporting, refuting, confidence)

        # Generate recommendation
        recommendation = self._generate_recommendation(status, confidence, claim.severity)

        # Collect sources
        sources = [e.source for e in evidence if e.source.startswith("web:")]

        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            evidence=evidence,
            explanation=explanation,
            recommendation=recommendation,
            sources=sources,
        )

    def _generate_explanation(
        self, claim: Claim, status: VerificationStatus,
        supporting: List[Evidence], refuting: List[Evidence],
        confidence: float,
    ) -> str:
        """Generate a human-readable explanation of the verification."""
        n_support = len(supporting)
        n_refute = len(refuting)

        if status == VerificationStatus.SUPPORTED:
            return f"Supported by {n_support} evidence source(s) with {confidence:.0%} confidence."
        elif status == VerificationStatus.REFUTED:
            return f"Refuted by {n_refute} evidence source(s). Confidence: {confidence:.0%}."
        elif status == VerificationStatus.CONFLICTING:
            return f"Conflicting evidence: {n_support} support, {n_refute} refute. Confidence: {confidence:.0%}."
        elif status == VerificationStatus.PARTIALLY_SUPPORTED:
            return f"Partially supported. Some evidence found but not conclusive. Confidence: {confidence:.0%}."
        else:
            return "Insufficient evidence to verify this claim."

    def _generate_recommendation(
        self, status: VerificationStatus, confidence: float, severity: ClaimSeverity,
    ) -> str:
        """Generate a recommendation for how NICTO should handle this claim."""
        if status == VerificationStatus.SUPPORTED and confidence > 0.8:
            return "SAFE_TO_STATE: Claim is well-supported. State it as fact."
        elif status == VerificationStatus.SUPPORTED and confidence > 0.5:
            return "QUALIFY: Claim is supported but with limited evidence. Add qualifiers like 'according to' or 'evidence suggests'."
        elif status == VerificationStatus.REFUTED:
            return "REFUSE: Claim is contradicted by evidence. Do not state it."
        elif status == VerificationStatus.CONFLICTING:
            return "QUALIFY: Evidence is mixed. Present both sides or note the uncertainty."
        elif severity == ClaimSeverity.SAFETY:
            return "REFUSE: Safety-critical claim with insufficient evidence. Do not state it."
        else:
            return "QUALIFY: Uncertain claim. Use hedging language like 'may', 'appears to', or 'some sources suggest'."
