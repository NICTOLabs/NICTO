"""
NICTO AI - Text-to-Speech
Convert text to natural-sounding speech.

Supports:
- Local: piper, coqui-tts, espeak
- Cloud: OpenAI TTS API, Google Text-to-Speech
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpeechResult:
    """Result from text-to-speech synthesis"""
    audio_data: bytes
    format: str = "wav"  # wav, mp3, ogg
    duration_ms: float = 0.0
    engine: str = "unknown"
    sample_rate: int = 22050


class TextToSpeech:
    """
    Text-to-Speech engine for NICTO voice system.

    Generates natural-sounding speech from text.

    Usage:
        tts = TextToSpeech()
        result = tts.synthesize("Hello, how can I help you?")
        # result.audio_data contains the audio bytes
    """

    def __init__(self, engine: str = "auto", voice: str = "default"):
        """
        Args:
            engine: TTS engine ("piper", "coqui", "cloud", "auto")
            voice: Voice name/style
        """
        self.engine = engine
        self.voice = voice
        self._available_engine = self._detect_engine()
        logger.info("TTS initialized with engine: %s", self._available_engine)

    def synthesize(self, text: str, voice: Optional[str] = None) -> SpeechResult:
        """
        Convert text to speech.

        Args:
            text: Text to speak
            voice: Override default voice

        Returns:
            SpeechResult with audio data
        """
        if not text.strip():
            return SpeechResult(audio_data=b"", engine="none")

        voice_name = voice or self.voice

        if self._available_engine == "piper":
            return self._synthesize_piper(text, voice_name)
        elif self._available_engine == "coqui":
            return self._synthesize_coqui(text, voice_name)
        elif self._available_engine == "cloud":
            return self._synthesize_cloud(text, voice_name)
        elif self._available_engine == "espeak":
            return self._synthesize_espeak(text)
        else:
            logger.warning("No TTS engine available")
            return SpeechResult(audio_data=b"", engine="none")

    def synthesize_to_file(self, text: str, output_path: str, voice: Optional[str] = None) -> bool:
        """Synthesize speech and save to file."""
        result = self.synthesize(text, voice)
        if result.audio_data:
            with open(output_path, "wb") as f:
                f.write(result.audio_data)
            return True
        return False

    def _detect_engine(self) -> str:
        """Detect the best available TTS engine."""
        if self.engine != "auto":
            return self.engine

        # Try local engines
        import shutil
        if shutil.which("piper"):
            return "piper"

        try:
            from TTS.api import TTS
            return "coqui"
        except ImportError:
            pass

        if shutil.which("espeak"):
            return "espeak"

        if os.environ.get("OPENAI_API_KEY"):
            return "cloud"

        logger.debug("No TTS engine available. Install piper or coqui-tts.")
        return "none"

    def _synthesize_piper(self, text: str, voice: str) -> SpeechResult:
        """Synthesize using piper (local, fast)."""
        import subprocess
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            cmd = ["piper", "--model", voice, "--output_file", temp_path]
            proc = subprocess.run(cmd, input=text.encode(), capture_output=True, timeout=30)

            if proc.returncode == 0 and os.path.exists(temp_path):
                with open(temp_path, "rb") as f:
                    audio_data = f.read()
                return SpeechResult(audio_data=audio_data, engine="piper")
            else:
                logger.error("Piper failed: %s", proc.stderr.decode())
                return SpeechResult(audio_data=b"", engine="piper")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _synthesize_coqui(self, text: str, voice: str) -> SpeechResult:
        """Synthesize using coqui-tts (local, high quality)."""
        try:
            from TTS.api import TTS
            import tempfile
            import os

            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            try:
                tts.tts_to_file(text=text, file_path=temp_path)
                with open(temp_path, "rb") as f:
                    audio_data = f.read()
                return SpeechResult(audio_data=audio_data, engine="coqui")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            logger.error("Coqui TTS failed: %s", e)
            return SpeechResult(audio_data=b"", engine="coqui")

    def _synthesize_cloud(self, text: str, voice: str) -> SpeechResult:
        """Synthesize using OpenAI TTS API (cloud)."""
        import requests

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return SpeechResult(audio_data=b"", engine="cloud")

        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "tts-1",
                "input": text,
                "voice": voice if voice != "default" else "alloy",
            },
            timeout=30,
        )

        if response.status_code == 200:
            return SpeechResult(
                audio_data=response.content,
                format="mp3",
                engine="cloud",
            )
        else:
            logger.error("Cloud TTS failed: %s", response.text)
            return SpeechResult(audio_data=b"", engine="cloud")

    def _synthesize_espeak(self, text: str) -> SpeechResult:
        """Synthesize using espeak (basic, always available)."""
        import subprocess

        try:
            proc = subprocess.run(
                ["espeak", "-w", "/dev/stdout", text],
                capture_output=True,
                timeout=10,
            )
            if proc.returncode == 0:
                return SpeechResult(audio_data=proc.stdout, engine="espeak")
        except Exception as e:
            logger.error("espeak failed: %s", e)

        return SpeechResult(audio_data=b"", engine="espeak")
