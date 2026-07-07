import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("NICTO VOICE ENGINE - 5 TESTS")
print("=" * 50)

# TEST 1
print("\nTEST 1: Backend Interface + EchoBackend")
from nicto_ai.voice.backend_interface import LLMBackend, EchoBackend, Message, CompletionResult
backend = EchoBackend()
assert backend.is_available() and backend.name == "echo"
result = backend.complete([Message(role="user", content="Hello")])
assert result.text == "Echo: Hello"
print("  PASS")

# TEST 2
print("\nTEST 2: NICTOBackend loads checkpoint + generates")
from nicto_ai.voice.backend_interface import NICTOBackend
ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nicto_model_final.pt")
t0 = time.time()
nicto = NICTOBackend(checkpoint_path=ckpt)
print("  Load: {:.1f}s".format(time.time() - t0))
assert nicto.is_available()
t0 = time.time()
result = nicto.complete([Message(role="user", content="Hi")], max_tokens=10)
elapsed = time.time() - t0
print("  Gen 10 tokens: {:.1f}s".format(elapsed))
assert result.text is not None
assert elapsed < 15, "Generation too slow: {:.1f}s".format(elapsed)
print("  PASS")

# TEST 3
print("\nTEST 3: Multi-turn conversation")
msgs = [
    Message(role="user", content="What is 2+2?"),
    Message(role="assistant", content="4"),
    Message(role="user", content="And 3+3?"),
]
t0 = time.time()
result = nicto.complete(msgs, max_tokens=10, system_prompt="Math tutor.")
print("  Response in {:.1f}s".format(time.time() - t0))
assert result.text is not None
print("  PASS")

# TEST 4
print("\nTEST 4: VoiceAgent with NICTO backend")
from nicto_ai.voice.agent_loop import VoiceAgent, AgentConfig
# Use echo backend for VoiceAgent to avoid re-loading model
config = AgentConfig(llm_backend="echo", enable_code_execution=False)
agent = VoiceAgent(config=config)
# Replace backend with our pre-loaded NICTO backend
agent._backend = nicto
assert type(agent._backend).__name__ == "NICTOBackend"
t0 = time.time()
resp = agent.process_text("Say hello", max_tokens=10)
print("  Response in {:.1f}s".format(time.time() - t0))
assert resp is not None
agent.clear_history()
assert agent.state.total_turns == 0
print("  PASS")

# TEST 5
print("\nTEST 5: Sandbox executor security")
from nicto_ai.voice.executor import SandboxExecutor
sandbox = SandboxExecutor(timeout=5)

r = sandbox.execute('import os; os.system("echo HACKED")', "python")
assert not r.success or "HACKED" not in r.output, "os module should be blocked"
print("  os blocked: OK")

r = sandbox.execute('import subprocess; subprocess.run("echo HACKED", shell=True)', "python")
assert not r.success or "HACKED" not in r.output, "subprocess should be blocked"
print("  subprocess blocked: OK")

r = sandbox.execute("del /f /q C:\\important.txt", "bash")
assert not r.success
assert "Blocked" in r.error
print("  bash del blocked: OK")

r = sandbox.execute("print('hello')", "python")
assert r.success and "hello" in r.output
print("  safe python: OK")

r = sandbox.execute("import sys; sys.stdout.write('hello')", "python")
assert r.success and "hello" in r.output
print("  safe python stdout: OK")
print("  PASS")

print("\n" + "=" * 50)
print("ALL 5 TESTS PASSED")
print("=" * 50)
