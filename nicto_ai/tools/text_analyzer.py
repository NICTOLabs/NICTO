"""
NICTO AI - Text Analyzer Tool
Analyze text for sentiment, entities, keywords, readability, and more.
"""

import re
import math
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class TextAnalyzerTool(Tool):
    """
    Comprehensive text analysis engine.

    Capabilities:
    - Sentiment analysis (positive/negative/neutral)
    - Named entity recognition
    - Keyword extraction
    - Readability scoring (Flesch-Kincaid)
    - Text statistics (word count, sentence length)
    - Language detection
    - Tone analysis
    """

    name = "text_analyzer"
    description = "Analyze text: sentiment, entities, keywords, readability, tone, language detection, and statistics."
    parameters = [
        ToolParameter(name="text", type="string", description="Text to analyze", required=True),
        ToolParameter(name="analysis", type="string", description="Type of analysis", required=False, default="all", enum=[
            "all", "sentiment", "entities", "keywords", "readability",
            "statistics", "tone", "language",
        ]),
        ToolParameter(name="extract_entities", type="boolean", description="Extract named entities", required=False, default=True),
        ToolParameter(name="sentiment_granularity", type="string", description="Sentiment detail level", required=False, default="document", enum=[
            "document", "sentence",
        ]),
    ]
    tags = ["text", "analysis", "nlp", "sentiment", "entities"]
    timeout_seconds = 30.0

    # Sentiment lexicons
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", "outstanding",
        "perfect", "best", "love", "happy", "joy", "beautiful", "awesome", "nice",
        "brilliant", "superb", "magnificent", "delightful", "pleasant", "enjoy",
        "success", "win", "positive", "benefit", "advantage", "improve", "better",
        "fast", "efficient", "effective", "reliable", "trust", "quality", "premium",
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst", "hate", "ugly", "poor",
        "fail", "failure", "error", "problem", "issue", "bug", "crash", "broken",
        "slow", "difficult", "hard", "impossible", "never", "waste", "loss",
        "negative", "disadvantage", "harmful", "dangerous", "risk", "threat",
        "disappoint", "frustrate", "annoy", "anger", "sad", "unhappy", "miserable",
    }

    # Entity patterns
    ENTITY_PATTERNS = {
        "PERSON": [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+\b',
        ],
        "ORG": [
            r'\b(?:Apple|Google|Microsoft|Amazon|Meta|Tesla|Netflix|OpenAI|IBM|Intel|Oracle)\b',
            r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co|Group|Company)\b',
        ],
        "LOCATION": [
            r'\b(?:New York|San Francisco|London|Tokyo|Paris|Berlin|Sydney|Toronto)\b',
            r'\b[A-Z][a-z]+(?:ton|burg|land|ville|field|wood)\b',
        ],
        "DATE": [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2},? \d{4}\b',
        ],
        "MONEY": [
            r'\$[\d,]+(?:\.\d{2})?',
            r'€[\d,]+(?:\.\d{2})?',
            r'£[\d,]+(?:\.\d{2})?',
        ],
        "EMAIL": [
            r'\b[\w.-]+@[\w.-]+\.\w+\b',
        ],
        "URL": [
            r'https?://\S+',
            r'www\.\S+',
        ],
    }

    # Tone indicators
    TONE_INDICATORS = {
        "formal": ["therefore", "furthermore", "consequently", "moreover", "hence", "thus", "accordingly"],
        "informal": ["yeah", "gonna", "wanna", "gotta", "kinda", "sorta", "lol", "btw", "imo"],
        "technical": ["algorithm", "implementation", "optimization", "framework", "architecture", "protocol"],
        "persuasive": ["must", "should", "need", "essential", "critical", "imperative", "vital"],
        "academic": ["hypothesis", "methodology", "empirical", "theoretical", "significant", "analysis"],
    }

    def _execute(self, text: str, analysis: str = "all", extract_entities: bool = True,
                 sentiment_granularity: str = "document") -> ToolResult:

        if not text or not text.strip():
            return ToolResult(success=False, error="Text is empty")

        result = {
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "analysis": analysis,
        }

        if analysis in ("all", "statistics"):
            result["statistics"] = self._compute_statistics(text)

        if analysis in ("all", "sentiment"):
            result["sentiment"] = self._analyze_sentiment(text, sentiment_granularity)

        if analysis in ("all", "entities") and extract_entities:
            result["entities"] = self._extract_entities(text)

        if analysis in ("all", "keywords"):
            result["keywords"] = self._extract_keywords(text)

        if analysis in ("all", "readability"):
            result["readability"] = self._compute_readability(text)

        if analysis in ("all", "tone"):
            result["tone"] = self._analyze_tone(text)

        if analysis in ("all", "language"):
            result["language"] = self._detect_language(text)

        return ToolResult(success=True, output=result)

    def _compute_statistics(self, text: str) -> Dict:
        """Compute text statistics"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Word lengths
        word_lengths = [len(w) for w in words]

        return {
            "characters": len(text),
            "characters_no_spaces": len(text.replace(" ", "")),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len(paragraphs),
            "avg_word_length": round(sum(word_lengths) / len(word_lengths), 1) if word_lengths else 0,
            "avg_sentence_length": round(len(words) / len(sentences), 1) if sentences else 0,
            "longest_word": max(words, key=len) if words else "",
            "shortest_word": min(words, key=len) if words else "",
        }

    def _analyze_sentiment(self, text: str, granularity: str) -> Dict:
        """Analyze sentiment"""
        words = re.findall(r'\b\w+\b', text.lower())

        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total = len(words) if words else 1

        pos_ratio = pos_count / total
        neg_ratio = neg_count / total

        # Compound score (-1 to 1)
        compound = (pos_ratio - neg_ratio) / max(pos_ratio + neg_ratio, 0.001)

        # Classification
        if compound > 0.05:
            label = "positive"
        elif compound < -0.05:
            label = "negative"
        else:
            label = "neutral"

        result = {
            "label": label,
            "compound": round(compound, 3),
            "positive_ratio": round(pos_ratio, 3),
            "negative_ratio": round(neg_ratio, 3),
            "positive_words_found": [w for w in words if w in self.POSITIVE_WORDS][:5],
            "negative_words_found": [w for w in words if w in self.NEGATIVE_WORDS][:5],
        }

        if granularity == "sentence":
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            result["sentence_sentiments"] = []
            for sent in sentences[:10]:
                sent_words = re.findall(r'\b\w+\b', sent.lower())
                s_pos = sum(1 for w in sent_words if w in self.POSITIVE_WORDS)
                s_neg = sum(1 for w in sent_words if w in self.NEGATIVE_WORDS)
                s_compound = (s_pos - s_neg) / max(s_pos + s_neg, 0.001)
                result["sentence_sentiments"].append({
                    "text": sent[:100],
                    "compound": round(s_compound, 3),
                })

        return result

    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities"""
        entities = {}
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            found = set()
            for pattern in patterns:
                matches = re.findall(pattern, text)
                found.update(matches)
            if found:
                entities[entity_type] = list(found)[:10]
        return entities

    def _extract_keywords(self, text: str) -> List[Dict]:
        """Extract keywords with TF scoring"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
                     "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
                     "who", "did", "get", "got", "let", "say", "she", "too", "use"}
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        total = sum(word_freq.values()) if word_freq else 1
        keywords = []
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]:
            keywords.append({
                "word": word,
                "frequency": freq,
                "tf": round(freq / total, 3),
            })
        return keywords

    def _compute_readability(self, text: str) -> Dict:
        """Compute readability scores"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        syllable_count = sum(self._count_syllables(w) for w in words)

        # Flesch-Kincaid Grade Level
        if sentences and words:
            fk_grade = (0.39 * (len(words) / len(sentences)) +
                       11.8 * (syllable_count / len(words)) - 15.59)
        else:
            fk_grade = 0

        # Flesch Reading Ease
        if sentences and words:
            fk_ease = (206.835 - 1.015 * (len(words) / len(sentences)) -
                      84.6 * (syllable_count / len(words)))
        else:
            fk_ease = 0

        # Reading level
        if fk_ease >= 90:
            level = "Very Easy (5th grade)"
        elif fk_ease >= 80:
            level = "Easy (6th grade)"
        elif fk_ease >= 70:
            level = "Fairly Easy (7th grade)"
        elif fk_ease >= 60:
            level = "Standard (8th-9th grade)"
        elif fk_ease >= 50:
            level = "Fairly Difficult (10th-12th grade)"
        elif fk_ease >= 30:
            level = "Difficult (College level)"
        else:
            level = "Very Difficult (Graduate level)"

        return {
            "flesch_kincaid_grade": round(fk_grade, 1),
            "flesch_reading_ease": round(max(0, min(100, fk_ease)), 1),
            "reading_level": level,
            "avg_syllables_per_word": round(syllable_count / len(words), 2) if words else 0,
        }

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count"""
        word = word.lower()
        if len(word) <= 3:
            return 1

        vowels = "aeiou"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        if word.endswith('e'):
            count -= 1
        return max(1, count)

    def _analyze_tone(self, text: str) -> Dict:
        """Analyze writing tone"""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        total_words = len(re.findall(r'\b\w+\b', text.lower()))

        tone_scores = {}
        for tone, indicators in self.TONE_INDICATORS.items():
            matches = words.intersection(set(indicators))
            score = len(matches) / len(indicators) if indicators else 0
            tone_scores[tone] = round(score, 3)

        # Determine primary tone
        primary_tone = max(tone_scores, key=tone_scores.get) if tone_scores else "neutral"

        # Check formality markers
        formal_markers = len(re.findall(r'\b(?:therefore|furthermore|consequently|moreover)\b', text.lower()))
        informal_markers = len(re.findall(r'\b(?:yeah|gonna|wanna|lol|btw)\b', text.lower()))

        if formal_markers > informal_markers:
            formality = "formal"
        elif informal_markers > formal_markers:
            formality = "informal"
        else:
            formality = "neutral"

        return {
            "primary_tone": primary_tone,
            "tone_scores": tone_scores,
            "formality": formality,
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
        }

    def _detect_language(self, text: str) -> Dict:
        """Simple language detection based on common words"""
        # Very basic - just detect English vs non-English patterns
        common_english = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but"}
        words = set(re.findall(r'\b\w+\b', text.lower()))
        english_score = len(words.intersection(common_english)) / len(common_english)

        return {
            "detected_language": "english" if english_score > 0.3 else "unknown",
            "confidence": round(english_score, 2),
            "note": "Basic detection only. For accurate detection, use a language detection library.",
        }
