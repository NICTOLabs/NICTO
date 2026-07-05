"""
NICTO AI - Claim Extractor
Extracts factual claims from text for verification.

Every statement NICTO makes is decomposed into atomic claims
that can be independently verified against sources.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    FACTUAL = "factual"           # "The Earth orbits the Sun"
    STATISTICAL = "statistical"   # "75% of users prefer X"
    TEMPORAL = "temporal"         # "Python was released in 1991"
    CAUSAL = "causal"             # "Rain causes flooding"
    COMPARATIVE = "comparative"   # "X is faster than Y"
    DEFINITIONAL = "definitional" # "AI is defined as..."
    NUMERICAL = "numerical"       # "There are 50 states"
    QUOTE = "quote"               # Direct quotations
    PREDICTION = "prediction"     # "X will happen by 2030"
    OPINION = "opinion"           # "X is the best" (subjective)


class ClaimSeverity(Enum):
    LOW = "low"           # Minor detail, not critical
    MEDIUM = "medium"     # Important but not safety-critical
    HIGH = "high"         # Critical fact, must be correct
    SAFETY = "safety"     # Safety-critical, must never be wrong


@dataclass
class Claim:
    """A single factual claim extracted from text"""
    text: str
    claim_type: ClaimType
    severity: ClaimSeverity
    confidence: float  # How confident we are this is a verifiable claim
    context: str  # Surrounding text for context
    position: Tuple[int, int]  # Start/end char positions in original text
    entities: List[str] = field(default_factory=list)  # Named entities
    is_negated: bool = False  # "X is NOT Y"
    is_quantified: bool = False  # Contains numbers/percentages
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"Claim({self.claim_type.value}: {self.text[:60]}...)"


class ClaimExtractor:
    """
    Extracts atomic factual claims from text.

    Uses pattern-based extraction combined with linguistic heuristics
    to identify statements that can be verified against external sources.

    The extractor is intentionally conservative - it prefers to extract
    too few claims rather than too many, to avoid false positives.
    """

    # Patterns that indicate factual claims
    FACTUAL_PATTERNS = [
        # Definitive statements
        (r"(?:^|\.\s+)([A-Z][^.]*?\s+is\s+a\s+[^.]*\.)", ClaimType.DEFINITIONAL, ClaimSeverity.MEDIUM),
        (r"(?:^|\.\s+)([A-Z][^.]*?\s+are\s+a\s+[^.]*\.)", ClaimType.DEFINITIONAL, ClaimSeverity.MEDIUM),
        # Causal claims
        (r"([^.]*?\s+causes?\s+[^.]*\.)", ClaimType.CAUSAL, ClaimSeverity.HIGH),
        (r"([^.]*?\s+leads?\s+to\s+[^.]*\.)", ClaimType.CAUSAL, ClaimSeverity.HIGH),
        (r"([^.]*?\s+results?\s+in\s+[^.]*\.)", ClaimType.CAUSAL, ClaimSeverity.HIGH),
        # Comparative claims
        (r"([^.]*?\s+(?:is|are)\s+(?:faster|slower|better|worse|more|less)\s+than\s+[^.]*\.)", ClaimType.COMPARATIVE, ClaimSeverity.MEDIUM),
        # Numerical claims
        (r"([^.]*?\s+(?:approximately|about|roughly|around|nearly|over|more than|less than)\s+\d[\d,.]*\s*%[^.]*\.)", ClaimType.STATISTICAL, ClaimSeverity.HIGH),
        (r"([^.]*?\s+(?:exactly|precisely)\s+\d[\d,.]*\s+[^.]*\.)", ClaimType.NUMERICAL, ClaimSeverity.HIGH),
        # Temporal claims
        (r"([^.]*?\s+(?:in|on|since|before|after|during)\s+(?:19|20)\d{2}[^.]*\.)", ClaimType.TEMPORAL, ClaimSeverity.MEDIUM),
        # Negated claims (important to catch)
        (r"([^.]*?\s+is\s+not\s+[^.]*\.)", ClaimType.FACTUAL, ClaimSeverity.HIGH),
        (r"([^.]*?\s+does\s+not\s+[^.]*\.)", ClaimType.FACTUAL, ClaimSeverity.HIGH),
        (r"([^.]*?\s+do\s+not\s+[^.]*\.)", ClaimType.FACTUAL, ClaimSeverity.HIGH),
        # General factual statements
        (r"(?:^|\.\s+)([A-Z][^.]*?\s+(?:has|have|had|was|were|can|will|should)\s+[^.]*\.)", ClaimType.FACTUAL, ClaimSeverity.MEDIUM),
        # Subject-verb-object (general factual)
        (r"(?:^|\.\s+)([A-Z][a-z]+\s+\w+\s+\w+[^.]*\.)", ClaimType.FACTUAL, ClaimSeverity.LOW),
    ]

    # Patterns that indicate opinions (NOT facts)
    OPINION_PATTERNS = [
        r"(?:I think|I believe|in my opinion|arguably|perhaps|maybe|possibly)",
        r"(?:the best|the worst|the most important)",
        r"(?:should|ought to|must be)",
        r"(?:beautiful|ugly|amazing|terrible|wonderful|awful)",
    ]

    # Named entity patterns
    ENTITY_PATTERNS = [
        (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", "PERSON_OR_ORG"),
        (r"\b(\d{4})\b", "YEAR"),
        (r"\b(\d[\d,.]*\s*%)\b", "PERCENTAGE"),
        (r"\b(\$[\d,.]+)\b", "MONEY"),
        (r"\b(\d[\d,.]*\s+(?:million|billion|trillion))\b", "LARGE_NUMBER"),
    ]

    def __init__(self, min_claim_length: int = 10, max_claim_length: int = 500):
        self.min_claim_length = min_claim_length
        self.max_claim_length = max_claim_length
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), ct, cs)
            for p, ct, cs in self.FACTUAL_PATTERNS
        ]
        self._opinion_patterns = [re.compile(p, re.IGNORECASE) for p in self.OPINION_PATTERNS]
        self._entity_patterns = [(re.compile(p), label) for p, label in self.ENTITY_PATTERNS]

    def extract(self, text: str) -> List[Claim]:
        """
        Extract all factual claims from text.

        Args:
            text: Input text to analyze

        Returns:
            List of Claim objects, sorted by severity (highest first)
        """
        if not text or not text.strip():
            return []

        claims = []
        sentences = self._split_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < self.min_claim_length:
                continue
            if len(sentence) > self.max_claim_length:
                # Split long sentences
                sub_sentences = self._split_long_sentence(sentence)
                for sub in sub_sentences:
                    claims.extend(self._extract_from_sentence(sub, text))
            else:
                claims.extend(self._extract_from_sentence(sentence, text))

        # Sort by severity (safety > high > medium > low)
        severity_order = {
            ClaimSeverity.SAFETY: 0,
            ClaimSeverity.HIGH: 1,
            ClaimSeverity.MEDIUM: 2,
            ClaimSeverity.LOW: 3,
        }
        claims.sort(key=lambda c: severity_order.get(c.severity, 4))

        # Deduplicate
        claims = self._deduplicate(claims)

        logger.debug("Extracted %d claims from %d characters", len(claims), len(text))
        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Jr|Sr|St|vs|etc|e\.g|i\.e)\.", r"\1<DOT>", text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.replace("<DOT>", ".") for s in sentences if s.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into shorter clauses."""
        # Split on semicolons, colons, or "and"/"but"/"or" followed by a capital letter
        parts = re.split(r"[;:]|(?:\s+(?:and|but|or)\s+(?=[A-Z]))", sentence)
        return [p.strip() for p in parts if p.strip()]

    def _extract_from_sentence(self, sentence: str, full_text: str) -> List[Claim]:
        """Extract claims from a single sentence."""
        claims = []

        # Check if this is an opinion
        is_opinion = any(p.search(sentence) for p in self._opinion_patterns)
        if is_opinion:
            # Still extract but mark as opinion
            claim = self._make_claim(sentence, ClaimType.OPINION, ClaimSeverity.LOW, full_text)
            if claim:
                claims.append(claim)
            return claims

        # Try each pattern
        for pattern, claim_type, severity in self._compiled_patterns:
            match = pattern.search(sentence)
            if match:
                claim_text = match.group(1) if match.lastindex else match.group(0)
                claim_text = claim_text.strip()

                if len(claim_text) < self.min_claim_length:
                    continue

                claim = Claim(
                    text=claim_text,
                    claim_type=claim_type,
                    severity=severity,
                    confidence=self._estimate_claim_confidence(claim_text),
                    context=sentence,
                    position=(full_text.find(claim_text), full_text.find(claim_text) + len(claim_text)),
                    entities=self._extract_entities(claim_text),
                    is_negated=self._check_negation(claim_text),
                    is_quantified=self._check_quantification(claim_text),
                )
                claims.append(claim)
                break  # One claim per sentence (most specific pattern wins)

        # If no pattern matched but sentence is a definitive statement
        if not claims and self._is_definitive_statement(sentence):
            claim = self._make_claim(sentence, ClaimType.FACTUAL, ClaimSeverity.MEDIUM, full_text)
            if claim:
                claims.append(claim)

        return claims

    def _make_claim(self, text: str, claim_type: ClaimType, severity: ClaimSeverity, full_text: str) -> Optional[Claim]:
        """Create a Claim object."""
        text = text.strip()
        if len(text) < self.min_claim_length or len(text) > self.max_claim_length:
            return None
        return Claim(
            text=text,
            claim_type=claim_type,
            severity=severity,
            confidence=self._estimate_claim_confidence(text),
            context=text,
            position=(full_text.find(text), full_text.find(text) + len(text)),
            entities=self._extract_entities(text),
            is_negated=self._check_negation(text),
            is_quantified=self._check_quantification(text),
        )

    def _estimate_claim_confidence(self, text: str) -> float:
        """Estimate how confident we are that this is a verifiable claim."""
        confidence = 0.5

        # Higher confidence for specific claims
        if self._check_quantification(text):
            confidence += 0.2
        if self._extract_entities(text):
            confidence += 0.1
        if any(w in text.lower() for w in ["is", "are", "was", "were", "has", "have"]):
            confidence += 0.1
        if text.endswith("."):
            confidence += 0.05

        return min(confidence, 1.0)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        for pattern, label in self._entity_patterns:
            for match in pattern.finditer(text):
                entities.append(f"{label}:{match.group(1)}")
        return entities

    def _check_negation(self, text: str) -> bool:
        """Check if the claim is negated."""
        negation_words = ["not", "never", "neither", "nobody", "nothing",
                         "nowhere", "nor", "cannot", "can't", "don't",
                         "doesn't", "didn't", "won't", "wouldn't", "shouldn't"]
        words = text.lower().split()
        return any(w in negation_words for w in words)

    def _check_quantification(self, text: str) -> bool:
        """Check if the claim contains numerical quantification."""
        return bool(re.search(r"\d+[%xX]|\$\d+|\d+\s*(?:million|billion|trillion|percent|%)", text))

    def _is_definitive_statement(self, text: str) -> bool:
        """Check if a sentence is a definitive factual statement."""
        # Starts with capital, ends with period
        if not text[0].isupper():
            return False
        if not text.endswith("."):
            return False
        # Definitive verbs including common action verbs
        definitive_verbs = [
            "is", "are", "was", "were", "has", "have", "had",
            "can", "could", "will", "would", "shall", "should",
            "orbits", "orbit", "contains", "measures", "weighs",
            "produces", "creates", "causes", "leads", "results",
            "requires", "uses", "provides", "includes", "occurs",
            "exists", "lies", "sits", "stands", "belongs",
        ]
        words = text.lower().split()
        return any(v in words for v in definitive_verbs)

    def _deduplicate(self, claims: List[Claim]) -> List[Claim]:
        """Remove duplicate claims."""
        seen = set()
        unique = []
        for claim in claims:
            # Normalize for dedup
            key = claim.text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(claim)
        return unique
