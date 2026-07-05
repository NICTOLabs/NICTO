"""
NICTO AI - LLM Backend Interface
Abstract interface for LLM backends.

The agent_loop.py uses this interface to communicate with
LLM providers without being coupled to any specific API.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """A chat message"""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Result from LLM completion"""
    text: str
    model: str = ""
    usage: Dict = field(default_factory=dict)  # tokens in/out
    finish_reason: str = "stop"
    metadata: Dict = field(default_factory=dict)


class LLMBackend(ABC):
    """
    Abstract LLM backend interface.

    Implement this interface to connect NICTO's voice system
    to any LLM provider (Claude, GPT-4, NICTO's own model, etc.).

    Example:
        class ClaudeBackend(LLMBackend):
            def complete(self, messages, **kwargs):
                # Call Anthropic API
                return CompletionResult(text=response)
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> CompletionResult:
        """
        Generate a completion from a list of messages.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt (optional)

        Returns:
            CompletionResult with generated text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and configured"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name"""
        pass


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API backend"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model

    def complete(self, messages, temperature=0.7, max_tokens=1024, system_prompt=None, **kwargs):
        import requests
        import os

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return CompletionResult(text="", finish_reason="error")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role in ("user", "assistant"):
                api_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                text = data.get("content", [{}])[0].get("text", "")
                return CompletionResult(
                    text=text,
                    model=data.get("model", self.model),
                    usage=data.get("usage", {}),
                )
            else:
                return CompletionResult(text="", finish_reason="error")
        except Exception as e:
            return CompletionResult(text="", finish_reason=f"error: {e}")

    def is_available(self):
        import os
        return bool(self.api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def name(self):
        return "claude"


class NICTOBackend(LLMBackend):
    """NICTO's own model backend (stub - wired to NICTOModel when checkpoint ready)"""

    def __init__(self, model=None):
        self._model = model

    def complete(self, messages, temperature=0.7, max_tokens=1024, system_prompt=None, **kwargs):
        if self._model is None:
            return CompletionResult(
                text="[NICTO model not loaded. Connect a trained checkpoint to use NICTOBackend.]",
                finish_reason="stub",
            )

        # When model is available, this would:
        # 1. Tokenize messages
        # 2. Run model.generate()
        # 3. Decode and return
        return CompletionResult(text="[NICTOBackend: model connected but generation not implemented yet]")

    def is_available(self):
        return self._model is not None

    @property
    def name(self):
        return "nicto"


class EchoBackend(LLMBackend):
    """Simple echo backend for testing"""

    def complete(self, messages, **kwargs):
        if messages:
            last = messages[-1]
            return CompletionResult(text=f"Echo: {last.content}")
        return CompletionResult(text="")

    def is_available(self):
        return True

    @property
    def name(self):
        return "echo"
