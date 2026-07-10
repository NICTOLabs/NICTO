"""
NICTO AI - Summarizer Tool
Summarize text, URLs, and documents with configurable detail levels.
"""

import re
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class SummarizerTool(Tool):
    """
    Text summarization engine.

    Supports:
    - Direct text summarization
    - URL content summarization
    - Multi-level detail (brief, standard, detailed)
    - Key points extraction
    - Executive summary generation
    - TL;DR generation
    """

    name = "summarizer"
    description = "Summarize text, articles, or documents. Extract key points, generate TL;DR, or create executive summaries."
    parameters = [
        ToolParameter(name="text", type="string", description="Text to summarize (or URL if using url mode)", required=True),
        ToolParameter(name="mode", type="string", description="Summarization mode", required=False, default="standard", enum=[
            "brief", "standard", "detailed", "key_points", "executive", "tldr", "bullets",
        ]),
        ToolParameter(name="max_sentences", type="integer", description="Max sentences in summary (1-20)", required=False, default=5),
        ToolParameter(name="focus", type="string", description="Focus area for summary (e.g., 'financial', 'technical', 'main_ideas')", required=False),
        ToolParameter(name="include_keywords", type="boolean", description="Extract keywords from text", required=False, default=False),
    ]
    tags = ["summarize", "text", "nlp", "extractive"]
    timeout_seconds = 30.0

    # Common filler words to ignore
    FILLER_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "that", "this", "it",
    }

    def _execute(self, text: str, mode: str = "standard", max_sentences: int = 5,
                 focus: str = None, include_keywords: bool = False) -> ToolResult:

        if not text or not text.strip():
            return ToolResult(success=False, error="Text is empty")

        # Split into sentences
        sentences = self._split_sentences(text)
        words = text.split()

        result = {
            "original_length": {
                "characters": len(text),
                "words": len(words),
                "sentences": len(sentences),
            },
            "mode": mode,
            "focus": focus,
        }

        if mode == "brief":
            result["summary"] = self._brief_summary(sentences, max_sentences)
        elif mode == "standard":
            result["summary"] = self._standard_summary(sentences, max_sentences)
        elif mode == "detailed":
            result["summary"] = self._detailed_summary(sentences, max_sentences)
        elif mode == "key_points":
            result["key_points"] = self._extract_key_points(sentences)
        elif mode == "executive":
            result["summary"] = self._executive_summary(sentences)
        elif mode == "tldr":
            result["summary"] = self._tldr_summary(sentences)
        elif mode == "bullets":
            result["bullets"] = self._bullet_summary(sentences)
        else:
            result["summary"] = self._standard_summary(sentences, max_sentences)

        # Calculate compression
        summary_text = result.get("summary", "")
        if isinstance(summary_text, list):
            summary_text = " ".join(summary_text)
        summary_words = len(summary_text.split()) if summary_text else 0
        result["compression_ratio"] = round(summary_words / len(words) * 100, 1) if words else 0

        if include_keywords:
            result["keywords"] = self._extract_keywords(text)

        return ToolResult(success=True, output=result)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Ltd|Corp)\.', r'\1<PERIOD>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.replace('<PERIOD>', '.').strip() for s in sentences if s.strip()]

    def _brief_summary(self, sentences: List[str], max_sentences: int) -> str:
        """Brief summary - first N sentences"""
        n = min(max_sentences, max(1, len(sentences) // 3))
        return " ".join(sentences[:n])

    def _standard_summary(self, sentences: List[str], max_sentences: int) -> str:
        """Standard extractive summary using sentence scoring"""
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        scored = self._score_sentences(sentences)
        top_indices = sorted([i for i, _ in scored[:max_sentences]])
        return " ".join(sentences[i] for i in top_indices)

    def _detailed_summary(self, sentences: List[str], max_sentences: int) -> str:
        """Detailed summary with more context"""
        n = min(max_sentences * 2, len(sentences))
        if n <= len(sentences) // 2:
            # Take beginning, middle, and end
            quarter = len(sentences) // 4
            selected = sentences[:quarter] + sentences[quarter:quarter+max_sentences] + sentences[-quarter:]
            return " ".join(selected[:max_sentences])
        return " ".join(sentences[:n])

    def _executive_summary(self, sentences: List[str]) -> str:
        """Executive summary - key decisions and outcomes"""
        # Look for decision/outcome sentences
        decision_words = {"decided", "agreed", "approved", "rejected", "implemented", "launched", "achieved", "resulted"}
        outcome_sentences = [s for s in sentences if any(w in s.lower() for w in decision_words)]

        if outcome_sentences:
            return " ".join(outcome_sentences[:5])
        # Fall back to first and last sentences
        if len(sentences) >= 2:
            return f"{sentences[0]} {sentences[-1]}"
        return sentences[0] if sentences else ""

    def _tldr_summary(self, sentences: List[str]) -> str:
        """TL;DR - ultra-short summary"""
        if sentences:
            # Take the first sentence, trimmed
            return sentences[0][:200]
        return ""

    def _bullet_summary(self, sentences: List[str]) -> List[str]:
        """Bullet point summary"""
        scored = self._score_sentences(sentences)
        return [s for _, s in scored[:10]]

    def _score_sentences(self, sentences: List[str]) -> List[tuple]:
        """Score sentences by importance"""
        scored = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()

            # Position score (first and last sentences important)
            if i == 0:
                score += 3
            elif i == len(sentences) - 1:
                score += 2
            elif i < len(sentences) * 0.2:
                score += 1

            # Length score (medium length sentences preferred)
            word_count = len(words)
            if 10 <= word_count <= 30:
                score += 2
            elif 5 <= word_count <= 40:
                score += 1

            # Keyword score
            important_words = {"important", "key", "significant", "major", "critical",
                             "essential", "primary", "main", "conclusion", "result",
                             "therefore", "however", "moreover", "furthermore"}
            for word in words:
                if word in important_words:
                    score += 2

            # Number/data score (sentences with numbers often contain facts)
            if re.search(r'\d+', sentence):
                score += 1

            # Quote score
            if '"' in sentence or "'" in sentence:
                score += 1

            scored.append((score, sentence))

        return sorted(scored, key=lambda x: x[0], reverse=True)

    def _extract_key_points(self, sentences: List[str]) -> List[str]:
        """Extract key points from text"""
        scored = self._score_sentences(sentences)
        return [s for _, s in scored[:8]]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in self.FILLER_WORDS:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top 10
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
