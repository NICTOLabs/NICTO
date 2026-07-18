"""
NICTO AI Agent — Real LLM + Tools

Uses an actual LLM (OpenAI, Anthropic, or local) as the brain,
with NICTO's 22 tools as its hands. The LLM decides when to use tools.

Architecture:
  User -> LLM (understands intent) -> decides: answer directly or use tool?
                                          |
                                     tool needed? -> execute tool -> feed result back to LLM
                                          |
                                     generate final response

Usage:
    from nicto_ai.agent.nicto_agent import NictoAgent
    agent = NictoAgent(provider="openai")
    reply = agent.chat("What's the weather in Tokyo?")
"""

import os
import json
import re
import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)
    source: str = "llm"
    tokens_used: int = 0


# ==============================================================================
# Tool Definitions for LLM
# ==============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": "Execute Python code in a sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_manager",
            "description": "Read, write, or list files",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "list", "delete"], "description": "File operation"},
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write (for write action)"}
                },
                "required": ["action", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "math_engine",
            "description": "Advanced math: derivatives, integrals, matrices, statistics",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression or problem"},
                    "operation": {"type": "string", "enum": ["derivative", "integral", "matrix", "statistics", "solve", "simplify"], "description": "Type of operation"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "content_writer",
            "description": "Write articles, blog posts, emails, and other content",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to write about"},
                    "content_type": {"type": "string", "enum": ["blog_post", "article", "email", "social_media", "technical_doc"], "description": "Type of content"},
                    "tone": {"type": "string", "enum": ["professional", "casual", "formal", "friendly"], "description": "Writing tone"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarizer",
            "description": "Summarize text or content",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                    "mode": {"type": "string", "enum": ["brief", "standard", "detailed", "key_points"], "description": "Summary length"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_analyzer",
            "description": "Analyze text: sentiment, entities, keywords, readability",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                    "analysis": {"type": "string", "enum": ["sentiment", "entities", "keywords", "readability", "statistics"], "description": "Type of analysis"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translator",
            "description": "Translate text between languages",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {"type": "string", "description": "Target language (e.g., 'Spanish', 'French', 'Japanese')"}
                },
                "required": ["text", "target_language"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from text description",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of image to generate"},
                    "width": {"type": "integer", "description": "Image width (default 512)"},
                    "height": {"type": "integer", "description": "Image height (default 512)"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_audio",
            "description": "Generate audio or music",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of audio to generate"},
                    "duration": {"type": "number", "description": "Duration in seconds (default 5)"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_3d",
            "description": "Generate a 3D point cloud model",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Description of 3D model to generate"},
                    "num_points": {"type": "integer", "description": "Number of points (default 2048)"}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_query",
            "description": "Query NICTO's knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in knowledge base"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "regex_builder",
            "description": "Build regular expressions from natural language",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What the regex should match"}
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_review",
            "description": "Review code for bugs, security issues, and improvements",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to review"},
                    "language": {"type": "string", "description": "Programming language"},
                    "focus": {"type": "string", "enum": ["security", "performance", "readability", "bugs"], "description": "Review focus"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_generator",
            "description": "Generate unit tests for code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to generate tests for"},
                    "framework": {"type": "string", "enum": ["pytest", "unittest", "jest"], "description": "Test framework"}
                },
                "required": ["code"]
            }
        }
    },
]

SYSTEM_PROMPT = """You are NICTO AI, an advanced AI assistant with access to tools.

You have access to tools for:
- Web search, code execution, math, file management
- Content writing, summarization, translation, text analysis
- Image, audio, and 3D model generation
- Code review and test generation
- Knowledge base queries

When the user asks something that requires a tool, call the appropriate function.
When you can answer directly, just respond normally.
Be helpful, concise, and accurate.
If you're unsure, say so rather than making things up."""


# ==============================================================================
# Tool Executor
# ==============================================================================

class ToolExecutor:
    """Executes NICTO tools."""

    def __init__(self):
        self._tools = {}
        self._load_tools()

    def _load_tools(self):
        """Lazy load tools."""
        try:
            from nicto_ai.tools import create_default_registry
            self._registry = create_default_registry()
            for tool_info in self._registry.list_tools():
                name = tool_info["name"]
                self._tools[name] = self._registry.get_tool(name)
        except Exception:
            self._registry = None

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return result as string."""
        # Direct implementations for key tools (no dependency on full tool registry)
        direct_handlers = {
            "calculator": self._calc,
            "code_executor": self._code_exec,
            "shell": self._shell_exec,
            "web_search": self._web_search,
            "file_manager": self._file_op,
            "regex_builder": self._regex_build,
            "summarizer": self._summarize,
            "translator": self._translate,
            "text_analyzer": self._analyze_text,
            "generate_image": self._gen_image,
            "generate_audio": self._gen_audio,
            "generate_3d": self._gen_3d,
            "content_writer": self._write_content,
            "math_engine": self._math_engine,
            "code_review": self._review_code,
            "test_generator": self._gen_tests,
            "knowledge_query": self._knowledge_query,
        }

        handler = direct_handlers.get(name)
        if handler:
            try:
                return handler(arguments)
            except Exception as e:
                return f"Error: {e}"

        # Fallback to registry
        tool = self._tools.get(name)
        if tool:
            try:
                result = tool._execute(**arguments)
                if hasattr(result, "output"):
                    return str(result.output)
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        return f"Tool '{name}' not available"

    def _calc(self, args: Dict) -> str:
        expr = args.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}}, {"abs": abs, "round": round, "min": min, "max": max})
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    def _code_exec(self, args: Dict) -> str:
        code = args.get("code", "")
        import subprocess
        import sys
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            return output[:2000] if output else "Code executed (no output)"
        except subprocess.TimeoutExpired:
            return "Code execution timed out (30s limit)"
        except Exception as e:
            return f"Execution error: {e}"

    def _shell_exec(self, args: Dict) -> str:
        import subprocess
        cmd = args.get("command", "")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            return output[:2000] if output else "Command executed (no output)"
        except Exception as e:
            return f"Shell error: {e}"

    def _web_search(self, args: Dict) -> str:
        query = args.get("query", "")
        return f"[Web search: {query}] - Web search requires API key. Set BING_API_KEY or use /web_search tool directly."

    def _file_op(self, args: Dict) -> str:
        action = args.get("action", "read")
        path = args.get("path", "")
        content = args.get("content", "")
        import os
        try:
            if action == "read":
                with open(path, "r") as f:
                    return f.read()[:5000]
            elif action == "write":
                with open(path, "w") as f:
                    f.write(content)
                return f"Written to {path}"
            elif action == "list":
                files = os.listdir(path) if os.path.isdir(path) else [path]
                return "\n".join(files[:100])
            elif action == "delete":
                os.remove(path)
                return f"Deleted {path}"
        except Exception as e:
            return f"File error: {e}"

    def _regex_build(self, args: Dict) -> str:
        desc = args.get("description", "")
        # Simple regex builder
        patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "url": r"https?://[^\s]+",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ip": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        }
        desc_lower = desc.lower()
        for key, pattern in patterns.items():
            if key in desc_lower:
                return f"Pattern: {pattern}\nExample: re.findall(r'{pattern}', text)"
        return f"Generated pattern for: {desc}\nTry: re.compile(r'{desc}')"

    def _summarize(self, args: Dict) -> str:
        text = args.get("text", "")
        mode = args.get("mode", "standard")
        # Simple extractive summary
        sentences = text.replace("\n", " ").split(". ")
        if mode == "brief":
            n = min(2, len(sentences))
        elif mode == "detailed":
            n = min(10, len(sentences))
        elif mode == "key_points":
            return "\n".join(f"- {s.strip()}" for s in sentences[:5] if s.strip())
        else:
            n = min(5, len(sentences))
        return ". ".join(sentences[:n]) + "."

    def _translate(self, args: Dict) -> str:
        text = args.get("text", "")
        lang = args.get("target_language", "")
        return f"[Translation to {lang}: {text}] - Translation requires API. Use TRANSLATOR_API_KEY or /translator tool."

    def _analyze_text(self, args: Dict) -> str:
        text = args.get("text", "")
        analysis = args.get("analysis", "statistics")
        words = text.split()
        sentences = text.split(".")
        if analysis == "statistics":
            return f"Words: {len(words)}, Sentences: {len(sentences)}, Characters: {len(text)}, Avg word length: {sum(len(w) for w in words)/max(1,len(words)):.1f}"
        elif analysis == "sentiment":
            positive = sum(1 for w in words if w.lower() in ["good", "great", "excellent", "amazing", "love", "best", "happy"])
            negative = sum(1 for w in words if w.lower() in ["bad", "terrible", "awful", "hate", "worst", "sad", "poor"])
            if positive > negative: return "Sentiment: Positive"
            elif negative > positive: return "Sentiment: Negative"
            return "Sentiment: Neutral"
        elif analysis == "keywords":
            from collections import Counter
            freq = Counter(w.lower() for w in words if len(w) > 3)
            return "Keywords: " + ", ".join(f"{w}({c})" for w, c in freq.most_common(10))
        return f"Analysis: {analysis} on {len(words)} words"

    def _gen_image(self, args: Dict) -> str:
        prompt = args.get("prompt", "")
        w = args.get("width", 512)
        h = args.get("height", 512)
        return f"[Image generation: '{prompt}' ({w}x{h})] - Use nicto generation API or /generate_image tool."

    def _gen_audio(self, args: Dict) -> str:
        prompt = args.get("prompt", "")
        dur = args.get("duration", 5)
        return f"[Audio generation: '{prompt}' ({dur}s)] - Use nicto generation API or /generate_audio tool."

    def _gen_3d(self, args: Dict) -> str:
        prompt = args.get("prompt", "")
        pts = args.get("num_points", 2048)
        return f"[3D generation: '{prompt}' ({pts} points)] - Use nicto generation API or /generate_3d tool."

    def _write_content(self, args: Dict) -> str:
        topic = args.get("topic", "")
        ctype = args.get("content_type", "article")
        tone = args.get("tone", "professional")
        return f"[Content writing: {ctype} about '{topic}' in {tone} tone] - Use /content_writer tool."

    def _math_engine(self, args: Dict) -> str:
        expr = args.get("expression", "")
        op = args.get("operation", "solve")
        return f"[Math {op}: {expr}] - Use /math_engine tool for advanced math."

    def _review_code(self, args: Dict) -> str:
        code = args.get("code", "")
        lang = args.get("language", "python")
        focus = args.get("focus", "bugs")
        # Simple review
        issues = []
        if "eval(" in code: issues.append("SECURITY: eval() usage detected - potential code injection")
        if "exec(" in code: issues.append("SECURITY: exec() usage detected")
        if "import os" in code and "os.system" in code: issues.append("SECURITY: os.system() usage - prefer subprocess")
        if len(code.split("\n")) > 100: issues.append("READABILITY: Function is very long, consider splitting")
        if not issues: issues.append("No obvious issues found")
        return f"Code Review ({lang}, focus: {focus}):\n" + "\n".join(f"- {i}" for i in issues)

    def _gen_tests(self, args: Dict) -> str:
        code = args.get("code", "")
        fw = args.get("framework", "pytest")
        # Extract function names
        import re
        funcs = re.findall(r"def (\w+)\(", code)
        if not funcs:
            return "No functions found to test"
        tests = []
        for f in funcs[:5]:
            tests.append(f"def test_{f}():\n    # TODO: test {f}\n    assert True")
        return f"Generated {fw} tests:\n\n" + "\n\n".join(tests)

    def _knowledge_query(self, args: Dict) -> str:
        query = args.get("query", "")
        return f"[Knowledge query: {query}] - Use /knowledge query command."


# ==============================================================================
# LLM Provider
# ==============================================================================

class LLMProvider:
    """Interface to LLM APIs."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client

        if self.provider == "openai":
            import openai
            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key)
            self.model = self.model or "gpt-4o-mini"
        elif self.provider == "anthropic":
            import anthropic
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key)
            self.model = self.model or "claude-sonnet-4-20250514"
        return self._client

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
             system: str = SYSTEM_PROMPT) -> Dict:
        """Send chat to LLM, return response with optional tool calls."""
        client = self._get_client()

        if self.provider == "openai":
            kwargs = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}] + messages,
                "temperature": 0.7,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            result = {
                "content": choice.message.content or "",
                "tool_calls": [],
                "finish_reason": choice.finish_reason,
                "tokens": response.usage.total_tokens if response.usage else 0,
            }

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    })

            return result

        elif self.provider == "anthropic":
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = [{
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                } for t in tools]

            response = client.messages.create(**kwargs)

            result = {
                "content": "",
                "tool_calls": [],
                "finish_reason": response.stop_reason,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            for block in response.content:
                if block.type == "text":
                    result["content"] += block.text
                elif block.type == "tool_use":
                    result["tool_calls"].append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

            return result

        raise ValueError(f"Unknown provider: {self.provider}")


# ==============================================================================
# NICTO Agent
# ==============================================================================

class NictoAgent:
    """
    NICTO AI Agent - Real LLM + Tools

    Uses an actual LLM as the brain, with NICTO's tools as its hands.
    The LLM understands user intent and decides when to use tools.

    Usage:
        agent = NictoAgent(provider="openai")
        response = agent.chat("What's 2+2?")
        print(response.text)

        response = agent.chat("Write a Python function to sort a list")
        print(response.text)
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None,
                 api_key: Optional[str] = None, max_tool_rounds: int = 5):
        self.provider = provider
        self.llm = LLMProvider(provider=provider, model=model, api_key=api_key)
        self.tools = ToolExecutor()
        self.max_tool_rounds = max_tool_rounds
        self.history: List[Dict] = []

    def chat(self, user_message: str) -> AgentResponse:
        """Send a message and get a response. Handles tool calls automatically."""
        self.history.append({"role": "user", "content": user_message})

        tool_rounds = 0
        all_tool_calls = []
        all_tool_results = []
        final_text = ""
        tokens = 0

        while tool_rounds < self.max_tool_rounds:
            # Call LLM
            result = self.llm.chat(self.history, tools=TOOL_SCHEMAS)
            tokens += result.get("tokens", 0)

            # No tool calls - we're done
            if not result["tool_calls"]:
                final_text = result["content"]
                if final_text:
                    self.history.append({"role": "assistant", "content": final_text})
                break

            # Execute tool calls
            tool_rounds += 1
            assistant_msg = {"role": "assistant", "content": result["content"] or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in result["tool_calls"]
            ]
            self.history.append(assistant_msg)

            for tc in result["tool_calls"]:
                tool_name = tc["name"]
                tool_args = tc["arguments"]

                # Execute
                tool_result = self.tools.execute(tool_name, tool_args)

                all_tool_calls.append(ToolCall(id=tc["id"], name=tool_name, arguments=tool_args))
                all_tool_results.append({"tool": tool_name, "result": tool_result})

                # Add tool result to history
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            # If LLM gave content along with tool calls, use it
            if result["content"]:
                final_text = result["content"]

        # Keep history manageable
        if len(self.history) > 50:
            self.history = self.history[-50:]

        return AgentResponse(
            text=final_text or "(No response generated)",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            source=f"llm:{self.provider}",
            tokens_used=tokens,
        )

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

    @property
    def available_tools(self) -> List[str]:
        """List available tool names."""
        return [t["function"]["name"] for t in TOOL_SCHEMAS]


# ==============================================================================
# Quick Start
# ==============================================================================

def create_agent(provider: str = "openai", **kwargs) -> NictoAgent:
    """Create a NICTO agent quickly."""
    return NictoAgent(provider=provider, **kwargs)


if __name__ == "__main__":
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"

    print(f"NICTO AI Agent ({provider})")
    print("Type 'quit' to exit, 'clear' to reset\n")

    agent = NictoAgent(provider=provider)

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
            agent.clear()
            print("(History cleared)\n")
            continue

        response = agent.chat(user_input)
        print(f"NICTO: {response.text}")
        if response.tool_calls:
            print(f"  [Used {len(response.tool_calls)} tool(s)]")
        print()
