"""
NICTO Chat System.

Multiple backends for conversational AI:
  - EchoBackend: echo input (testing)
  - RuleBackend: pattern-matching responses (no model needed)
  - APIBackend: connect to OpenAI/Anthropic/etc (needs API key)
  - LocalBackend: use NICTO's own trained model

Usage:
    from nicto_ai.chat import NictoChat
    chat = NictoChat(backend="rule")
    reply = chat.send("Hello!")
"""

import re
import random
import json
import os
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from chat."""
    text: str
    confidence: float = 1.0
    source: str = "backend"
    metadata: Dict = field(default_factory=dict)


# ==============================================================================
# Chat Backends
# ==============================================================================

class ChatBackend:
    """Base class for chat backends."""

    def reply(self, messages: List[ChatMessage]) -> ChatResponse:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "base"


class EchoBackend(ChatBackend):
    """Echo backend for testing."""

    def reply(self, messages: List[ChatMessage]) -> ChatResponse:
        if messages:
            last = messages[-1].content
            return ChatResponse(text=f"Echo: {last}", source="echo")
        return ChatResponse(text="")

    @property
    def name(self) -> str:
        return "echo"


class RuleBackend(ChatBackend):
    """Pattern-matching chatbot. No model needed."""

    def __init__(self):
        self.patterns = self._build_patterns()
        self.name_ = "rule"

    def _build_patterns(self) -> List[tuple]:
        """Build pattern -> response pairs."""
        return [
            # Greetings
            (r"\b(hi|hello|hey|howdy|greetings)\b",
             ["Hello! I'm NICTO. How can I help you?",
              "Hey there! What can I do for you?",
              "Hi! Ask me anything or try /tools to see my capabilities."]),

            # How are you
            (r"\b(how are you|how r u|how's it going|what's up)\b",
             ["I'm running great! All systems operational.",
              "Doing well, thanks for asking! What can I help with?",
              "All good here. Ready to help!"]),

            # What are you
            (r"\b(what are you|who are you|tell me about yourself)\b",
             ["I'm NICTO AI — a neural architecture with multi-modal generation, "
              "22 built-in tools, and a creativity engine. I can generate images, "
              "video, audio, 3D point clouds, and chat with you!",
              "I'm NICTO, an open-source AI with tool-aware chat, multi-modal "
              "generation, and iterative creativity. Try /tools to see what I can do!"]),

            # What can you do
            (r"\b(what can you do|your capabilities|help me|what do you know)\b",
             ["I can help with many things! Try these:\n"
              "  - /tools     -> See all 22 built-in tools\n"
              "  - /calculator -> Do math\n"
              "  - /web_search -> Search the web\n"
              "  - Just chat with me naturally!\n"
              "I can also generate images, video, audio, and 3D models."]),

            # Thanks
            (r"\b(thanks|thank you|thx|ty)\b",
             ["You're welcome! Anything else I can help with?",
              "Happy to help! Let me know if you need anything else.",
              "No problem! I'm here if you need more help."]),

            # Goodbye
            (r"\b(bye|goodbye|see you|later|exit|quit)\b",
             ["Goodbye! Have a great day!",
              "See you later! Come back anytime.",
              "Bye! It was nice chatting with you."]),

            # Time
            (r"\b(what time|current time|what's the time)\b",
             [f"It's {{time}}. But I don't have a clock — just joking!",
              "I don't track time, but you can check your system clock!"]),

            # Name
            (r"\b(what's your name|your name|name)\b",
             ["I'm NICTO AI! Named after the neural architecture I run on.",
              "NICTO — that's me! Nice to meet you."]),

            # Joke
            (r"\b(tell me a joke|joke|funny)\b",
             ["Why do programmers prefer dark mode? Because light attracts bugs!",
              "Why did the AI go to therapy? It had too many deep learning issues!",
              "What's a computer's favorite snack? Microchips!",
              "Why was the math book sad? Because it had too many problems!"]),

            # Weather
            (r"\b(weather|temperature|forecast)\b",
             ["I don't have access to weather data, but you can try /web_search for your local weather!",
              "I can't check the weather directly, but try searching the web with /web_search."]),

            # Meaning of life
            (r"\b(meaning of life|42|deep question)\b",
             ["42, according to Douglas Adams. But I think the real meaning is to keep learning!",
              "The answer is 42. The question is... what's the question?"]),

            # capabilities
            (r"\b(generate|create|make|build)\b.*\b(image|picture|photo)\b",
             ["I can generate images! Use the /generate_image tool or ask me to create something specific.",
              "Image generation is one of my skills! Try asking me to generate an image."]),

            (r"\b(generate|create|make)\b.*\b(video|animation)\b",
             ["I can generate video clips! Use the /generate_video tool.",
              "Video generation is available! Ask me to create a video."]),

            (r"\b(generate|create|make)\b.*\b(audio|music|sound)\b",
             ["I can generate audio and music! Use the /generate_audio tool.",
              "Audio generation is one of my multi-modal capabilities!"]),

            (r"\b(generate|create|make)\b.*\b(3d|three.d|point cloud|model)\b",
             ["I can generate 3D point clouds! Use the /generate_3d tool.",
              "3D generation is available! I can create point clouds from text."]),

            # Code help
            (r"\b(code|programming|python|javascript|debug|function)\b",
             ["I can help with code! Try:\n"
              "  - /code_executor -> Run Python code\n"
              "  - /code_review   -> Review code\n"
              "  - /test_generator -> Generate tests\n"
              "  Just paste your code and ask me about it!"]),

            # Math
            (r"\b(math|calculate|compute|solve|equation)\b",
             ["I can help with math! Try:\n"
              "  - /calculator 2+2 -> Simple calculations\n"
              "  - /math_engine    -> Advanced math (derivatives, matrices, stats)\n"
              "  Just ask me a math question!"]),

            # Search
            (r"\b(search|find|look up|google|web)\b",
             ["I can search the web! Try:\n"
              "  - /web_search query=your question\n"
              "I'll find the answer for you."]),

            # Default
            (r".*",
             ["I'm not sure how to help with that. Try /tools to see what I can do!",
              "Interesting! Can you tell me more? Or try /tools for my capabilities.",
              "I don't have a specific answer for that, but I'm learning! Try /help for commands."]),
        ]

    def reply(self, messages: List[ChatMessage]) -> ChatResponse:
        if not messages:
            return ChatResponse(text="Hello! I'm NICTO. How can I help you?")

        user_msg = messages[-1].content.lower().strip()

        for pattern, responses in self.patterns:
            if re.search(pattern, user_msg, re.IGNORECASE):
                response = random.choice(responses)
                # Handle time placeholder
                if "{time}" in response:
                    from datetime import datetime
                    response = response.replace("{time}", datetime.now().strftime("%H:%M"))
                return ChatResponse(text=response, source="rule")

        return ChatResponse(text="I'm here to help! Try /tools to see my capabilities.")

    @property
    def name(self) -> str:
        return "rule"


class APIBackend(ChatBackend):
    """Backend using external API (OpenAI, Anthropic, etc)."""

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None,
                 model: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
                self.model = self.model or "gpt-4o-mini"
            except ImportError:
                raise ImportError("pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                self.model = self.model or "claude-sonnet-4-20250514"
            except ImportError:
                raise ImportError("pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def reply(self, messages: List[ChatMessage]) -> ChatResponse:
        if not self.api_key:
            return ChatResponse(
                text=f"API key not set. Set {self.provider.upper()}_API_KEY environment variable.",
                confidence=0.0,
                source="api_error",
            )

        client = self._get_client()

        system_msg = "You are NICTO AI, a helpful multi-modal AI assistant with 22 built-in tools."
        chat_messages = [{"role": "system", "content": system_msg}]
        for msg in messages[-10:]:  # Last 10 messages
            if msg.role in ("user", "assistant"):
                chat_messages.append({"role": msg.role, "content": msg.content})

        try:
            if self.provider == "openai":
                response = client.chat.completions.create(
                    model=self.model,
                    messages=chat_messages,
                    max_tokens=1024,
                    temperature=0.7,
                )
                text = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_msg,
                    messages=[{"role": m["role"], "content": m["content"]}
                              for m in chat_messages if m["role"] != "system"],
                )
                text = response.content[0].text
            else:
                text = f"Unknown provider: {self.provider}"

            return ChatResponse(text=text, source=f"api:{self.provider}")
        except Exception as e:
            return ChatResponse(text=f"API error: {e}", confidence=0.0, source="api_error")

    @property
    def name(self) -> str:
        return f"api:{self.provider}"


# ==============================================================================
# Main Chat Interface
# ==============================================================================

class NictoChat:
    """
    NICTO Chat - conversational AI interface.

    Usage:
        chat = NictoChat(backend="rule")
        reply = chat.send("Hello!")
        print(reply.text)

        # Or with history
        chat.send("What's 2+2?")
        chat.send("And multiply by 3?")
        print(chat.get_history())
    """

    def __init__(self, backend: str = "rule", **kwargs):
        self.history: List[ChatMessage] = []
        self.max_history = 20

        if backend == "echo":
            self.backend = EchoBackend()
        elif backend == "rule":
            self.backend = RuleBackend()
        elif backend == "openai":
            self.backend = APIBackend(provider="openai", **kwargs)
        elif backend == "anthropic":
            self.backend = APIBackend(provider="anthropic", **kwargs)
        elif backend == "api":
            self.backend = APIBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use: echo, rule, openai, anthropic")

    def send(self, message: str) -> ChatResponse:
        """Send a message and get a reply."""
        import time

        # Add user message
        user_msg = ChatMessage(role="user", content=message, timestamp=time.time())
        self.history.append(user_msg)

        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Get reply
        response = self.backend.reply(self.history)

        # Add assistant reply to history
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.text,
            timestamp=time.time(),
            metadata={"source": response.source, "confidence": response.confidence},
        )
        self.history.append(assistant_msg)

        return response

    def get_history(self) -> List[Dict]:
        """Get chat history as dicts."""
        return [{"role": m.role, "content": m.content} for m in self.history]

    def clear(self):
        """Clear chat history."""
        self.history.clear()

    def set_system(self, prompt: str):
        """Set system prompt (for API backends)."""
        self.history.insert(0, ChatMessage(role="system", content=prompt))


# ==============================================================================
# CLI Chat
# ==============================================================================

def run_chat(backend: str = "rule", **kwargs):
    """Run interactive chat in terminal."""
    from datetime import datetime

    print("  _   _ ___ ___  ____ ___  ")
    print(" | \\ | |_ _/ _ \\/ ___/ _ \\ ")
    print(" |  \\| | | | | | |  | | | |")
    print(" | |\\  | | | |_| | |__| |_| |")
    print(" |_| \\_|___|\\___/\\____\\___/ ")
    print()
    print(f"  Backend: {backend}")
    print("  Type 'quit' to exit, 'clear' to reset history")
    print("  " + "=" * 40)
    print()

    chat = NictoChat(backend=backend, **kwargs)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            chat.clear()
            print("(History cleared)\n")
            continue

        response = chat.send(user_input)
        print(f"NICTO: {response.text}")
        if response.source != "rule":
            print(f"  [{response.source}]")
        print()


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "rule"
    run_chat(backend=backend)
