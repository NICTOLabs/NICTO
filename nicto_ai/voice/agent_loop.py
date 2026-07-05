"""
NICTO AI - Voice Agent Loop
Orchestrates: transcribe -> LLM -> parse actions -> execute -> speak

This is the main voice interaction loop that ties together:
- Speech-to-Text (transcribe user's voice)
- LLM Backend (generate response)
- Action Parser (detect code blocks, tool calls)
- Executor (run code in sandbox)
- Text-to-Speech (speak the response)
"""

import re
import logging
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .stt import SpeechToText, TranscriptionResult
from .tts import TextToSpeech, SpeechResult
from .executor import SandboxExecutor, ExecutionResult
from .backend_interface import LLMBackend, Message, CompletionResult, ClaudeBackend, NICTOBackend, EchoBackend

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the voice agent"""
    stt_engine: str = "auto"
    tts_engine: str = "auto"
    tts_voice: str = "default"
    llm_backend: str = "claude"  # "claude", "nicto", "echo"
    system_prompt: str = "You are NICTO, a helpful AI assistant. Respond concisely and accurately."
    max_history: int = 20  # Max conversation turns to remember
    enable_code_execution: bool = True
    enable_tool_use: bool = True
    sandbox_timeout: int = 10


@dataclass
class AgentState:
    """Current state of the voice agent"""
    is_listening: bool = False
    is_speaking: bool = False
    is_processing: bool = False
    conversation_history: List[Message] = field(default_factory=list)
    last_transcription: str = ""
    last_response: str = ""
    total_turns: int = 0


class VoiceAgent:
    """
    NICTO's Voice Agent - the main voice interaction loop.

    Flow:
    ┌──────────────┐
    │ User speaks   │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ STT          │ → text
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ LLM Backend  │ → response (may contain code blocks)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Action Parse │ → detect code, tool calls
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Executor     │ → run code in sandbox
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ TTS          │ → speak response
    └──────────────┘
    """

    def __init__(self, config: AgentConfig = None, tools=None):
        """
        Args:
            config: Agent configuration
            tools: ToolRegistry instance for tool use
        """
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.tools = tools

        # Initialize components
        self.stt = SpeechToText(engine=self.config.stt_engine)
        self.tts = TextToSpeech(engine=self.config.tts_engine, voice=self.config.tts_voice)
        self.executor = SandboxExecutor(timeout=self.config.sandbox_timeout)

        # LLM backend
        self._backend = self._create_backend()

        # Callbacks for UI integration
        self.on_transcription: Optional[Callable] = None
        self.on_response: Optional[Callable] = None
        self.on_execution: Optional[Callable] = None
        self.on_speech: Optional[Callable] = None

    def process_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Process audio input through the full pipeline.

        Args:
            audio_data: Raw audio bytes

        Returns:
            The spoken response text, or None if no response
        """
        self.state.is_processing = True
        start_time = time.time()

        try:
            # Step 1: Transcribe
            transcription = self.stt.transcribe(audio_data)
            if not transcription.text.strip():
                return None

            self.state.last_transcription = transcription.text
            if self.on_transcription:
                self.on_transcription(transcription.text)

            # Step 2: Generate response
            response = self._generate_response(transcription.text)
            self.state.last_response = response

            # Step 3: Parse and execute actions
            final_response = self._process_actions(response)

            # Step 4: Speak response
            if final_response.strip():
                self.tts.synthesize(final_response)

            self.state.total_turns += 1

            elapsed = (time.time() - start_time) * 1000
            logger.info("Voice loop completed in %.1fms", elapsed)

            return final_response

        except Exception as e:
            logger.error("Voice agent error: %s", e)
            error_msg = "I encountered an error processing that. Could you try again?"
            self.tts.synthesize(error_msg)
            return error_msg
        finally:
            self.state.is_processing = False

    def process_text(self, text: str) -> Optional[str]:
        """
        Process text input (for text-based interaction).

        Same as process_audio but skips STT.
        """
        if not text.strip():
            return None

        self.state.last_transcription = text
        self.state.is_processing = True

        try:
            response = self._generate_response(text)
            final_response = self._process_actions(response)
            self.state.last_response = final_response
            self.state.total_turns += 1
            return final_response
        finally:
            self.state.is_processing = False

    def _generate_response(self, user_input: str) -> str:
        """Generate LLM response from user input."""
        # Add user message to history
        self.state.conversation_history.append(Message(role="user", content=user_input))

        # Trim history
        if len(self.state.conversation_history) > self.config.max_history:
            self.state.conversation_history = self.state.conversation_history[-self.config.max_history:]

        # Generate completion
        result = self._backend.complete(
            messages=self.state.conversation_history,
            system_prompt=self.config.system_prompt,
        )

        # Add assistant response to history
        self.state.conversation_history.append(Message(role="assistant", content=result.text))

        return result.text

    def _process_actions(self, response: str) -> str:
        """Parse response for executable actions and run them."""
        if not self.config.enable_code_execution:
            return response

        # Check for code blocks
        code_blocks = re.findall(r"```(\w+)?\n(.*?)```", response, re.DOTALL)

        if not code_blocks:
            return response

        processed_response = response
        for lang, code in code_blocks:
            if lang in ("python", "py", ""):
                result = self.executor.execute(code, language="python")
                if self.on_execution:
                    self.on_execution(result)

                # Replace code block with execution result
                if result.success:
                    replacement = f"```\n{code.strip()}\n```\n**Output:**\n```\n{result.output.strip()}\n```"
                else:
                    replacement = f"```\n{code.strip()}\n```\n**Error:** {result.error}"

                processed_response = processed_response.replace(
                    f"```{lang}\n{code}```",
                    replacement,
                    1,
                )

        return processed_response

    def _create_backend(self) -> LLMBackend:
        """Create the configured LLM backend."""
        if self.config.llm_backend == "claude":
            return ClaudeBackend()
        elif self.config.llm_backend == "nicto":
            return NICTOBackend()
        elif self.config.llm_backend == "echo":
            return EchoBackend()
        else:
            logger.warning("Unknown backend '%s', using echo", self.config.llm_backend)
            return EchoBackend()

    def get_state(self) -> Dict:
        """Get current agent state."""
        return {
            "is_listening": self.state.is_listening,
            "is_speaking": self.state.is_speaking,
            "is_processing": self.state.is_processing,
            "total_turns": self.state.total_turns,
            "last_transcription": self.state.last_transcription,
            "last_response": self.state.last_response,
            "history_length": len(self.state.conversation_history),
        }

    def clear_history(self):
        """Clear conversation history."""
        self.state.conversation_history.clear()
        self.state.total_turns = 0

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.config.system_prompt = prompt
