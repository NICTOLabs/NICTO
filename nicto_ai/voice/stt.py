"""
NICTO AI - Speech-to-Text
Transcribe audio to text using local or cloud STT engines.

Supports:
- Local: whisper.cpp, faster-whisper
- Cloud: OpenAI Whisper API, Google Speech-to-Text
"""

import os
import logging
import tempfile
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""
    text: str
    language: str = "en"
    confidence: float = 0.0
    duration_ms: float = 0.0
    engine: str = "unknown"


class SpeechToText:
    """
    Speech-to-Text engine for NICTO voice system.

    Automatically selects the best available engine:
    1. Try local whisper.cpp/faster-whisper (fastest, no API needed)
    2. Fall back to cloud API if local not available

    Usage:
        stt = SpeechToText()
        result = stt.transcribe(audio_bytes)
        print(result.text)
    """

    def __init__(self, engine: str = "auto", language: str = "en"):
        """
        Args:
            engine: STT engine ("whisper", "faster_whisper", "cloud", "auto")
            language: Default language code
        """
        self.engine = engine
        self.language = language
        self._available_engine = self._detect_engine()
        logger.info("STT initialized with engine: %s", self._available_engine)

    def transcribe(self, audio_data: bytes, language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio bytes (WAV, MP3, etc.)
            language: Override default language

        Returns:
            TranscriptionResult with text and metadata
        """
        lang = language or self.language

        if not audio_data or len(audio_data) < 100:
            return TranscriptionResult(
                text="",
                language=lang,
                engine=self._available_engine,
                confidence=0.0,
            )

        if self._available_engine == "whisper":
            return self._transcribe_whisper(audio_data, lang)
        elif self._available_engine == "faster_whisper":
            return self._transcribe_faster_whisper(audio_data, lang)
        elif self._available_engine == "cloud":
            return self._transcribe_cloud(audio_data, lang)
        else:
            return TranscriptionResult(
                text="",
                language=lang,
                engine="none",
                confidence=0.0,
            )

    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe an audio file."""
        with open(file_path, "rb") as f:
            return self.transcribe(f.read(), language)

    def _detect_engine(self) -> str:
        """Detect the best available STT engine."""
        if self.engine != "auto":
            return self.engine

        # Try local engines first
        try:
            import whisper
            return "whisper"
        except ImportError:
            pass

        try:
            from faster_whisper import WhisperModel
            return "faster_whisper"
        except ImportError:
            pass

        # Fall back to cloud
        if os.environ.get("OPENAI_API_KEY"):
            return "cloud"

        logger.debug("No STT engine available. Install whisper or faster-whisper.")
        return "none"

    def _transcribe_whisper(self, audio_data: bytes, language: str) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper (local)."""
        import whisper
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            model = whisper.load_model("base")
            result = model.transcribe(temp_path, language=language)
            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language", language),
                confidence=sum(s["no_speech_prob"] for s in result["segments"]) / max(1, len(result["segments"])),
                engine="whisper",
            )
        finally:
            os.unlink(temp_path)

    def _transcribe_faster_whisper(self, audio_data: bytes, language: str) -> TranscriptionResult:
        """Transcribe using faster-whisper (local, faster)."""
        from faster_whisper import WhisperModel
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(temp_path, language=language)
            text = " ".join(s.text for s in segments)
            return TranscriptionResult(
                text=text.strip(),
                language=info.language,
                confidence=info.language_probability,
                engine="faster_whisper",
            )
        finally:
            os.unlink(temp_path)

    def _transcribe_cloud(self, audio_data: bytes, language: str) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API (cloud)."""
        import requests

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return TranscriptionResult(text="", engine="cloud", confidence=0.0)

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", audio_data, "audio/wav")},
            data={"model": "whisper-1", "language": language},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            return TranscriptionResult(
                text=data.get("text", ""),
                language=language,
                confidence=0.9,
                engine="cloud",
            )
        else:
            logger.error("Cloud STT failed: %s", response.text)
            return TranscriptionResult(text="", engine="cloud", confidence=0.0)
