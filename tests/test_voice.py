"""
Test NICTO AI Voice System
STT, TTS, Executor, Backend Interface, Agent Loop
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nicto_ai.voice.stt import SpeechToText, TranscriptionResult
from nicto_ai.voice.tts import TextToSpeech, SpeechResult
from nicto_ai.voice.executor import SandboxExecutor, ExecutionResult
from nicto_ai.voice.backend_interface import (
    LLMBackend, Message, CompletionResult,
    ClaudeBackend, NICTOBackend, EchoBackend,
)
from nicto_ai.voice.agent_loop import VoiceAgent, AgentConfig, AgentState


# ─── SpeechToText Tests ─────────────────────────────────────────

def test_stt_init():
    stt = SpeechToText()
    assert stt is not None
    print("  STT init: OK")

def test_stt_detect_engine():
    stt = SpeechToText()
    assert stt._available_engine in ("whisper", "faster_whisper", "cloud", "none")
    print("  STT detect engine: OK")

def test_stt_transcribe_empty():
    stt = SpeechToText()
    result = stt.transcribe(b"")
    assert isinstance(result, TranscriptionResult)
    print("  STT transcribe empty: OK")


# ─── TextToSpeech Tests ────────────────────────────────────────

def test_tts_init():
    tts = TextToSpeech()
    assert tts is not None
    print("  TTS init: OK")

def test_tts_detect_engine():
    tts = TextToSpeech()
    assert tts._available_engine in ("piper", "coqui", "espeak", "cloud", "none")
    print("  TTS detect engine: OK")

def test_tts_synthesize_empty():
    tts = TextToSpeech()
    result = tts.synthesize("")
    assert isinstance(result, SpeechResult)
    print("  TTS synthesize empty: OK")


# ─── SandboxExecutor Tests ──────────────────────────────────────

def test_executor_init():
    exec = SandboxExecutor()
    assert exec is not None
    print("  Executor init: OK")

def test_executor_python():
    exec = SandboxExecutor()
    result = exec.execute("print(2 + 2)", language="python")
    assert result.success
    assert "4" in result.output
    print("  Executor python: OK")

def test_executor_python_error():
    exec = SandboxExecutor()
    result = exec.execute("1/0", language="python")
    assert not result.success
    print("  Executor python error: OK")

def test_executor_bash():
    exec = SandboxExecutor()
    result = exec.execute("echo hello", language="bash")
    assert result.success
    assert "hello" in result.output
    print("  Executor bash: OK")

def test_executor_timeout():
    exec = SandboxExecutor(timeout=1)
    result = exec.execute("import time; time.sleep(10)", language="python")
    assert not result.success
    assert "timed out" in result.error.lower()
    print("  Executor timeout: OK")

def test_executor_unsupported():
    exec = SandboxExecutor()
    result = exec.execute("SELECT 1", language="sql")
    assert not result.success
    print("  Executor unsupported: OK")


# ─── Backend Interface Tests ────────────────────────────────────

def test_echo_backend():
    backend = EchoBackend()
    assert backend.is_available()
    assert backend.name == "echo"
    print("  Echo backend init: OK")

def test_echo_backend_complete():
    backend = EchoBackend()
    messages = [Message(role="user", content="Hello")]
    result = backend.complete(messages)
    assert isinstance(result, CompletionResult)
    assert "Echo" in result.text
    print("  Echo backend complete: OK")

def test_nicto_backend_stub():
    backend = NICTOBackend()
    assert not backend.is_available()
    messages = [Message(role="user", content="Hello")]
    result = backend.complete(messages)
    assert "not loaded" in result.text
    print("  NICTO backend stub: OK")

def test_claude_backend_init():
    backend = ClaudeBackend(api_key="test-key")
    assert backend.name == "claude"
    print("  Claude backend init: OK")


# ─── VoiceAgent Tests ───────────────────────────────────────────

def test_agent_init():
    config = AgentConfig(llm_backend="echo")
    agent = VoiceAgent(config=config)
    assert agent is not None
    print("  Agent init: OK")

def test_agent_process_text():
    config = AgentConfig(llm_backend="echo")
    agent = VoiceAgent(config=config)
    response = agent.process_text("Hello NICTO")
    assert response is not None
    assert "Echo" in response
    print("  Agent process text: OK")

def test_agent_state():
    config = AgentConfig(llm_backend="echo")
    agent = VoiceAgent(config=config)
    agent.process_text("Test message")
    state = agent.get_state()
    assert state["total_turns"] == 1
    assert state["last_transcription"] == "Test message"
    print("  Agent state: OK")

def test_agent_clear_history():
    config = AgentConfig(llm_backend="echo")
    agent = VoiceAgent(config=config)
    agent.process_text("Message 1")
    agent.process_text("Message 2")
    agent.clear_history()
    state = agent.get_state()
    assert state["history_length"] == 0
    assert state["total_turns"] == 0
    print("  Agent clear history: OK")

def test_agent_system_prompt():
    config = AgentConfig(llm_backend="echo")
    agent = VoiceAgent(config=config)
    agent.set_system_prompt("You are a test assistant.")
    assert agent.config.system_prompt == "You are a test assistant."
    print("  Agent system prompt: OK")

def test_agent_multi_turn():
    config = AgentConfig(llm_backend="echo")
    agent = VoiceAgent(config=config)
    agent.process_text("Hello")
    agent.process_text("How are you?")
    state = agent.get_state()
    assert state["total_turns"] == 2
    assert state["history_length"] == 4  # 2 user + 2 assistant
    print("  Agent multi-turn: OK")


# ─── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NICTO AI - Voice System Test Suite")
    print("=" * 60)

    print("\n--- SpeechToText ---")
    test_stt_init()
    test_stt_detect_engine()
    test_stt_transcribe_empty()

    print("\n--- TextToSpeech ---")
    test_tts_init()
    test_tts_detect_engine()
    test_tts_synthesize_empty()

    print("\n--- SandboxExecutor ---")
    test_executor_init()
    test_executor_python()
    test_executor_python_error()
    test_executor_bash()
    test_executor_timeout()
    test_executor_unsupported()

    print("\n--- Backend Interface ---")
    test_echo_backend()
    test_echo_backend_complete()
    test_nicto_backend_stub()
    test_claude_backend_init()

    print("\n--- VoiceAgent ---")
    test_agent_init()
    test_agent_process_text()
    test_agent_state()
    test_agent_clear_history()
    test_agent_system_prompt()
    test_agent_multi_turn()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
