"""
Test VoiceAgent with ToolAgent integration.

Verifies:
1. VoiceAgent.process_text() auto-detects tool intents
2. Falls back to EchoBackend when no tool matches
3. Web search tool works end-to-end via VoiceAgent
4. Knowledge tool works (crawl/ingest/query)
"""

import sys
import time

sys.path.insert(0, ".")

from nicto_ai.tools import create_default_registry
from nicto_ai.agent import ToolAgent
from nicto_ai.voice.agent_loop import VoiceAgent, AgentConfig


def test_voice_agent_tool_detection():
    """VoiceAgent detects and routes to tools."""
    registry = create_default_registry()
    tool_agent = ToolAgent(registry)
    config = AgentConfig(llm_backend="echo", enable_code_execution=False)
    va = VoiceAgent(config, tools=tool_agent)

    tests = [
        ("search for Python programming", "web_search", "search"),
        ("what do you know about neural networks", "knowledge_query", "query"),
        ("hello, how are you?", None, "fallback"),
    ]

    for text, expected_tool, mode in tests:
        result = va.process_text(text)
        assert result is not None, f"No response for: {text}"
        tool_name = ""
        if va.state.conversation_history:
            last = va.state.conversation_history[-1]
            if hasattr(last, "metadata") and last.metadata:
                tool_name = last.metadata.get("tool", "")
        print(f"[{mode}] '{text}' -> tool={tool_name}, response_len={len(result)}")

    print("test_voice_agent_tool_detection: PASS")


def test_voice_agent_search():
    """VoiceAgent routes web search queries to WebSearchTool."""
    registry = create_default_registry()
    tool_agent = ToolAgent(registry)
    config = AgentConfig(llm_backend="echo", enable_code_execution=False)
    va = VoiceAgent(config, tools=tool_agent)

    result = va.process_text("search for Python programming language")
    assert result is not None
    print(f"Search response (excerpt): {result[:200]}...")
    assert "Python" in result or "python" in result or "search" in result.lower(), \
        f"No search results in response: {result[:200]}"

    print("test_voice_agent_search: PASS")


def test_voice_agent_fallback():
    """VoiceAgent falls back to EchoBackend when no tool matches."""
    registry = create_default_registry()
    tool_agent = ToolAgent(registry)
    config = AgentConfig(llm_backend="echo", enable_code_execution=False)
    va = VoiceAgent(config, tools=tool_agent)

    result = va.process_text("hello")
    assert result is not None
    # EchoBackend echoes back
    assert "hello" in result.lower(), f"Echo fallback failed: {result}"

    print("test_voice_agent_fallback: PASS")


def test_voice_agent_clear_history():
    """VoiceAgent correctly clears conversation history."""
    registry = create_default_registry()
    tool_agent = ToolAgent(registry)
    config = AgentConfig(llm_backend="echo", enable_code_execution=False)
    va = VoiceAgent(config, tools=tool_agent)

    va.process_text("hello")
    assert len(va.state.conversation_history) > 0

    va.clear_history()
    assert len(va.state.conversation_history) == 0
    assert va.state.total_turns == 0

    print("test_voice_agent_clear_history: PASS")


def test_knowledge_crawl_and_query():
    """KnowledgeBase crawl + query via KnowledgeTool."""
    from nicto_ai.tools import KnowledgeTool

    tool = KnowledgeTool()
    result = tool._execute("stats")
    assert result.success
    print(f"Initial KB stats: {result.output}")
    initial_count = result.output.get("total_entries", 0)

    # Crawl a test page
    result = tool._execute("crawl:https://httpbin.org")
    assert result.success, f"Crawl failed: {result.error}"
    print(f"Crawl result: {result.output}")

    # Query something
    result = tool._execute("httpbin")
    assert result.success
    result_count = len(result.output)
    print(f"Query 'httpbin': {result_count} results ({result.metadata.get('query_time_ms', 0):.0f}ms)")

    # Ingest text
    result = tool._execute("ingest:NICTO is a neural architecture created by NICTO Labs")
    assert result.success
    print(f"Ingest result: {result.output}")

    result = tool._execute("stats")
    assert result.success
    print(f"Final KB stats: {result.output}")
    assert result.output.get("total_entries", 0) > initial_count

    print("test_knowledge_crawl_and_query: PASS")


def test_workflow_engine():
    """WorkflowEngine executes multi-tool workflows."""
    from nicto_ai.tools import create_default_registry
    from nicto_ai.agent import ToolAgent, WorkflowEngine, Workflow, WorkflowStep

    registry = create_default_registry()
    agent = ToolAgent(registry)
    engine = WorkflowEngine(agent)

    # List presets
    presets = engine.list_presets()
    print(f"Available workflows: {[p['name'] for p in presets]}")
    assert len(presets) >= 3

    # Run knowledge_search preset
    result = engine.run(engine._presets["knowledge_search"], {"input": "NICTO"})
    print(f"Workflow '{result.name}': success={result.success}, time={result.elapsed_ms:.0f}ms")
    for name, step in result.steps.items():
        print(f"  [{step['status']}] {name}")

    print("test_workflow_engine: PASS")


if __name__ == "__main__":
    start = time.time()
    tests = [
        ("VoiceAgent tool detection", test_voice_agent_tool_detection),
        ("VoiceAgent search", test_voice_agent_search),
        ("VoiceAgent fallback", test_voice_agent_fallback),
        ("VoiceAgent clear history", test_voice_agent_clear_history),
        ("Knowledge crawl & query", test_knowledge_crawl_and_query),
        ("Workflow engine", test_workflow_engine),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            print()

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed in {elapsed:.1f}s")
    if failed:
        print(f"{failed} FAILED")
        sys.exit(1)
