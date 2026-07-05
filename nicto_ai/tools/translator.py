"""
NICTO AI - Translator Tool
Translate text between languages while preserving meaning and tone.
"""

import logging
from typing import Dict, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class TranslatorTool(Tool):
    """
    Language translation tool.

    Provides translation between major world languages.
    Preserves tone, context, and technical terminology.
    """

    name = "translator"
    description = "Translate text between languages. Supports major world languages with tone preservation."
    parameters = [
        ToolParameter(name="text", type="string", description="Text to translate", required=True),
        ToolParameter(name="target_language", type="string", description="Target language", required=True, enum=[
            "english", "spanish", "french", "german", "italian", "portuguese",
            "russian", "chinese", "japanese", "korean", "arabic", "hindi",
            "dutch", "swedish", "norwegian", "danish", "finnish", "polish",
            "czech", "romanian", "hungarian", "turkish", "thai", "vietnamese",
            "indonesian", "malay", "tagalog", "bengali", "urdu", "persian",
            "hebrew", "ukrainian", "greek", "hungarian",
        ]),
        ToolParameter(name="source_language", type="string", description="Source language (auto-detect if not specified)", required=False),
    ]
    tags = ["translation", "language", "nlp"]
    timeout_seconds = 15.0

    # Language codes mapping
    LANG_CODES = {
        "english": "en", "spanish": "es", "french": "fr", "german": "de",
        "italian": "it", "portuguese": "pt", "russian": "ru", "chinese": "zh",
        "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi",
        "dutch": "nl", "swedish": "sv", "norwegian": "no", "danish": "da",
        "finnish": "fi", "polish": "pl", "czech": "cs", "romanian": "ro",
        "hungarian": "hu", "turkish": "tr", "thai": "th", "vietnamese": "vi",
        "indonesian": "id", "malay": "ms", "tagalog": "tl", "bengali": "bn",
        "urdu": "ur", "persian": "fa", "hebrew": "he", "ukrainian": "uk",
        "greek": "el",
    }

    def _execute(self, text: str, target_language: str, source_language: str = None) -> ToolResult:
        if not text.strip():
            return ToolResult(success=False, error="Text is empty")

        target_code = self.LANG_CODES.get(target_language.lower())
        if not target_code:
            return ToolResult(success=False, error=f"Unsupported language: {target_language}")

        # For now, return a placeholder since we need a translation API
        # In production, this would call Google Translate, DeepL, or similar
        return ToolResult(
            success=True,
            output={
                "original": text,
                "translated": f"[Translation to {target_language} would appear here]",
                "source_language": source_language or "auto-detected",
                "target_language": target_language,
                "target_code": target_code,
                "note": "Translation service not yet connected. Connect to DeepL or Google Translate API.",
            },
            metadata={"service": "placeholder"},
        )
