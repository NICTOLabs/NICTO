"""
NICTO AI - Source Attribution
Tracks and attributes all claims to their sources.

Every statement NICTO makes should be traceable to a source.
This module manages source tracking and generates citations.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A source of information"""
    id: str
    url: Optional[str] = None
    title: str = ""
    author: str = ""
    timestamp: Optional[str] = None
    source_type: str = "unknown"  # "web", "knowledge_base", "internal", "user"
    reliability: float = 0.5  # 0-1 reliability score
    access_date: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"Source({self.source_type}: {self.title[:40]})"


@dataclass
class Attribution:
    """Attribution of a claim to a source"""
    claim_text: str
    source: Source
    confidence: float  # How confident we are this source supports the claim
    quote: str = ""  # Exact quote from source (if available)
    page_section: str = ""  # Section of source (if applicable)


class SourceAttribution:
    """
    Manages source tracking and attribution for NICTO's outputs.

    Every factual claim NICTO makes should be attributed to at least
    one source. This module:
    1. Tracks which sources were used
    2. Links claims to their supporting sources
    3. Generates formatted citations
    4. Ranks source reliability

    This is NOT just for show - it's a critical anti-hallucination
    mechanism. If NICTO can't cite a source, it shouldn't make the claim.
    """

    def __init__(self):
        self._sources: Dict[str, Source] = {}
        self._attributions: List[Attribution] = []
        self._source_reliability: Dict[str, float] = {}

    def register_source(self, source: Source) -> str:
        """Register a new source and return its ID."""
        self._sources[source.id] = source
        return source.id

    def add_attribution(
        self,
        claim_text: str,
        source_id: str,
        confidence: float = 0.5,
        quote: str = "",
        page_section: str = "",
    ) -> Attribution:
        """Add an attribution linking a claim to a source."""
        source = self._sources.get(source_id)
        if source is None:
            logger.warning("Source %s not found, creating placeholder", source_id)
            source = Source(id=source_id, title=source_id, source_type="unknown")

        attr = Attribution(
            claim_text=claim_text,
            source=source,
            confidence=confidence,
            quote=quote,
            page_section=page_section,
        )
        self._attributions.append(attr)
        return attr

    def get_attributions_for_claim(self, claim_text: str) -> List[Attribution]:
        """Get all attributions for a specific claim."""
        return [
            a for a in self._attributions
            if self._claims_match(claim_text, a.claim_text)
        ]

    def get_all_sources(self) -> List[Source]:
        """Get all registered sources."""
        return list(self._sources.values())

    def get_reliability_score(self, source_id: str) -> float:
        """Get the reliability score for a source."""
        if source_id in self._source_reliability:
            return self._source_reliability[source_id]
        source = self._sources.get(source_id)
        return source.reliability if source else 0.0

    def update_reliability(self, source_id: str, was_accurate: bool):
        """Update source reliability based on accuracy."""
        current = self.get_reliability_score(source_id)
        if was_accurate:
            new_score = current + (1.0 - current) * 0.1  # Move toward 1.0
        else:
            new_score = current * 0.9  # Move toward 0.0
        self._source_reliability[source_id] = new_score

    def generate_citation(self, attribution: Attribution, style: str = "inline") -> str:
        """Generate a formatted citation."""
        source = attribution.source

        if style == "inline":
            if source.url:
                return f"[{source.title}]({source.url})"
            return f"[{source.title}]"

        elif style == "academic":
            parts = []
            if source.author:
                parts.append(source.author)
            if source.title:
                parts.append(f'"{source.title}"')
            if source.timestamp:
                parts.append(f"({source.timestamp})")
            if source.url:
                parts.append(source.url)
            return ", ".join(parts) if parts else "[Unknown source]"

        elif style == "numbered":
            idx = list(self._sources.keys()).index(source.id) + 1 if source.id in self._sources else 0
            return f"[{idx}]"

        return str(source)

    def generate_references_section(self, style: str = "inline") -> str:
        """Generate a formatted references section."""
        sources_used = set()
        for attr in self._attributions:
            sources_used.add(attr.source.id)

        if not sources_used:
            return ""

        lines = ["## References\n"]
        for i, source_id in enumerate(sorted(sources_used), 1):
            source = self._sources.get(source_id)
            if source:
                if style == "numbered":
                    lines.append(f"[{i}] {self.generate_citation(Attribution('', source, 1.0), 'academic')}")
                else:
                    lines.append(f"- {self.generate_citation(Attribution('', source, 1.0), 'academic')}")

        return "\n".join(lines)

    def get_attribution_stats(self) -> Dict:
        """Get statistics about attributions."""
        return {
            "total_sources": len(self._sources),
            "total_attributions": len(self._attributions),
            "sources_by_type": self._count_sources_by_type(),
            "avg_reliability": self._avg_reliability(),
        }

    def _count_sources_by_type(self) -> Dict[str, int]:
        """Count sources by type."""
        counts = {}
        for source in self._sources.values():
            counts[source.source_type] = counts.get(source.source_type, 0) + 1
        return counts

    def _avg_reliability(self) -> float:
        """Compute average reliability across all sources."""
        if not self._sources:
            return 0.0
        return sum(s.reliability for s in self._sources.values()) / len(self._sources)

    def _claims_match(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are similar enough to be considered the same."""
        # Simple word overlap matching
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.6
