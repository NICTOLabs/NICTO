"""
NICTO AI - Tool Agent
Orchestrates tools: auto-detect intent, invoke tools, format results.

The ToolAgent bridges natural language input to tool execution:
1. User says "write a blog post about Python"
2. ToolAgent detects content_writer tool
3. Extracts parameters (topic="Python", content_type="blog_post")
4. Invokes content_writer
5. Returns formatted result
"""

import re
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from nicto_ai.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from the tool agent"""
    text: str
    tool_name: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    execution_time_ms: float = 0.0
    auto_detected: bool = False
    success: bool = True
    error: Optional[str] = None


class ToolAgent:
    """
    Natural language tool invocation agent.

    Detects user intent and routes to the right tool.
    If no tool is detected, returns None so the caller
    can fall back to normal LLM chat.
    """

    # Intent -> tool mapping with parameter extraction patterns
    INTENTS = [
        {
            "keywords": ["write", "blog", "article", "post", "content", "draft"],
            "tool": "content_writer",
            "params": {
                "topic": r"(?:about|on|regarding)\s+([^.,!?]+)",
                "content_type": r"(blog\s*post|article|email|social\s*media|product\s*description|ad\s*copy|press\s*release|technical\s*doc|newsletter|speech|story|poem|resume|cover\s*letter|business\s*plan)",
                "tone": r"(professional|casual|formal|friendly|persuasive|humorous|technical|academic|creative|inspirational)",
            },
        },
        {
            "keywords": ["summarize", "summary", "tldr", "tl;dr", "key points", "brief"],
            "tool": "summarizer",
            "params": {
                "mode": r"(brief|standard|detailed|key_points|executive|tldr|bullets)",
            },
        },
        {
            "keywords": ["review", "code review", "analyze code", "check code", "review this", "review code", "scan code"],
            "tool": "code_review",
            "params": {
                "language": r"(python|javascript|typescript|java|c|cpp|go|rust|ruby|php)",
                "focus": r"(security|performance|readability|best_practices|bugs)",
            },
        },
        {
            "keywords": ["analyze", "sentiment", "readability", "text analysis"],
            "tool": "text_analyzer",
            "params": {
                "analysis": r"(sentiment|entities|keywords|readability|statistics|tone|language)",
            },
        },
        {
            "keywords": ["regex", "regular expression", "pattern match"],
            "tool": "regex_builder",
            "params": {
                "operation": r"(test|match|findall|explain|generate)",
            },
        },
        {
            "keywords": ["json", "validate json", "format json", "pretty print"],
            "tool": "json_builder",
            "params": {
                "operation": r"(validate|generate|transform|query|merge|format|minify)",
            },
        },
        {
            "keywords": ["markdown", "readme", "documentation", "docs"],
            "tool": "markdown_builder",
            "params": {
                "operation": r"(document|table|code_block|list|header|link|task_list|blockquote|readme)",
            },
        },
        {
            "keywords": ["test", "unit test", "pytest", "test generator"],
            "tool": "test_generator",
            "params": {
                "language": r"(python|javascript)",
                "test_type": r"(unit|edge_cases|integration|all)",
            },
        },
        {
            "keywords": ["calculate", "calculator", "math", "solve", "equation", "="],
            "tool": "calculator",
            "params": {},
        },
        {
            "keywords": ["translate", "translation", "in spanish", "in french", "in german"],
            "tool": "translator",
            "params": {
                "target_language": r"(english|spanish|french|german|italian|portuguese|russian|chinese|japanese|korean|arabic|hindi)",
            },
        },
        {
            "keywords": ["hash", "md5", "sha", "checksum", "crc"],
            "tool": "hash_tool",
            "params": {
                "algorithm": r"(md5|sha1|sha256|sha512|blake2b|crc32|adler32)",
            },
        },
        {
            "keywords": ["password", "generate password", "secure password"],
            "tool": "password_generator",
            "params": {
                "length": r"(\d+)\s*(?:char|character)",
            },
        },
        {
            "keywords": ["color", "palette", "color scheme", "hex color"],
            "tool": "color_palette",
            "params": {
                "scheme": r"(monochromatic|complementary|analogous|triadic|tetradic|random|pastel|vibrant|dark)",
            },
        },
        {
            "keywords": ["sql", "query", "select", "from", "database"],
            "tool": "text_to_sql",
            "params": {
                "dialect": r"(sqlite|postgresql|mysql)",
                "query_type": r"(select|insert|update|delete|create)",
            },
        },
        {
            "keywords": ["execute", "run code", "run python", "python code"],
            "tool": "code_executor",
            "params": {},
        },
        {
            "keywords": ["shell", "command", "terminal", "run command"],
            "tool": "shell",
            "params": {},
        },
        {
            "keywords": ["file", "read file", "write file", "list files", "directory"],
            "tool": "file_manager",
            "params": {
                "operation": r"(read|write|list|exists|info)",
            },
        },
        {
            "keywords": ["data analysis", "analyze csv", "analyze json", "correlation", "statistics on data"],
            "tool": "data_analysis",
            "params": {
                "operation": r"(summary|columns|correlations|quality|filter|group)",
            },
        },
        {
            "keywords": ["api", "http", "request", "fetch", "get url", "post to"],
            "tool": "api_caller",
            "params": {
                "method": r"(GET|POST|PUT|DELETE|PATCH)",
            },
        },
        {
            "keywords": ["knowledge base", "what do you know", "kb", "crawl", "search knowledge", "query knowledge"],
            "tool": "knowledge_query",
            "params": {
                "query": r"(?:about|on|for)\s+([^.,!?]+)",
                "source_filter": r"(web|document|manual)",
            },
        },
        {
            "keywords": ["search", "search web", "find online", "look up", "google", "bing", "internet search", "browse web"],
            "tool": "web_search",
            "params": {
                "query": r"(?:for|about|on)\s+([^.,!?]+)",
                "engine": r"(bing|google|duckduckgo)",
            },
        },
        {
            "keywords": ["simulate", "simulation", "hydrogen atom", "three body", "n-body", "n body",
                         "predator prey", "lotka volterra", "epidemiology", "sir model",
                         "run simulation", "physics sim", "particle sim"],
            "tool": "simulator",
            "params": {
                "query": r"simulate\s+(?:a\s+|an\s+)?(.+)",
            },
        },
    ]

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def process(self, text: str) -> Optional[AgentResponse]:
        """
        Process natural language text and invoke the appropriate tool.

        Returns None if no tool matches (caller should fall back to LLM).
        """
        start = time.time()
        text_lower = text.lower().strip()

        # Try exact tool invocation first: "tool_name(param=value, ...)"
        exact = self._try_exact_invocation(text_lower)
        if exact:
            exact.execution_time_ms = (time.time() - start) * 1000
            return exact

        # Auto-detect code blocks (route to code_review)
        code_block = self._detect_code_block(text)
        if code_block:
            tool = self.registry.get("code_review")
            if tool:
                params = {"code": code_block, "language": "python"}
                self._convert_param_types(tool, params)
                result = tool.execute(**params)
                elapsed = (time.time() - start) * 1000
                return AgentResponse(
                    text=self._format_result("code_review", result),
                    tool_name="code_review",
                    tool_result=result,
                    execution_time_ms=elapsed,
                    auto_detected=True,
                    success=result.success,
                    error=result.error if not result.success else None,
                )

        # Try intent matching
        for intent in self.INTENTS:
            keywords = intent["keywords"]
            if not any(kw in text_lower for kw in keywords):
                continue

            tool_name = intent["tool"]
            tool = self.registry.get(tool_name)
            if not tool:
                continue

            # Extract parameters from text
            params = self._extract_params(text, intent["params"])

            # For calculator, extract expression (strip prefix words)
            if tool_name == "calculator":
                expr = text
                for prefix in ["calculate", "calculator", "math", "solve", "what is", "what's"]:
                    if expr.lower().startswith(prefix):
                        expr = expr[len(prefix):].strip().lstrip(": ")
                params["expression"] = expr

            # For code_review, extract code from text
            if tool_name == "code_review":
                code = self._extract_code_block(text)
                if not code:
                    # Strip review prefix to get the code
                    code = text
                    for prefix in ["review this code:", "review code:", "review this:", "review:", "analyze code:", "check code:"]:
                        if code.lower().startswith(prefix):
                            code = code[len(prefix):].strip()
                            break
                params["code"] = code

            # For code_executor or shell, extract code block
            if tool_name in ("code_executor", "shell"):
                code = self._extract_code_block(text)
                if code:
                    params["code"] = code

            # For summarizer, use the full text as input
            if tool_name == "summarizer" and "text" not in params:
                params["text"] = text

            # For text_analyzer, use the full text
            if tool_name == "text_analyzer" and "text" not in params:
                params["text"] = text

            # For hash_tool, use the text
            if tool_name == "hash_tool" and "data" not in params:
                params["data"] = text

            # For color_palette, extract base_color
            if tool_name == "color_palette":
                color_match = re.search(r'(?:color|palette)\s+(\w+)', text_lower)
                if color_match:
                    params["base_color"] = color_match.group(1)

            # For text_to_sql, use the full text as query
            if tool_name == "text_to_sql" and "query" not in params:
                params["query"] = text

            # For api_caller, extract URL
            if tool_name == "api_caller":
                url_match = re.search(r'https?://[^\s,]+', text)
                if url_match:
                    params["url"] = url_match.group()

            # For web_search, extract query from remaining text
            if tool_name == "web_search" and "query" not in params:
                q = text
                for prefix in ["search the web for", "search web for", "search internet for", "search for",
                               "search the internet for", "search online for", "search", "look up",
                               "find online", "google", "bing", "browse web for", "browse"]:
                    if q.lower().startswith(prefix):
                        q = q[len(prefix):].strip().lstrip(":,. ")
                        break
                params["query"] = q

            # For knowledge_query, extract query from remaining text
            if tool_name == "knowledge_query" and "query" not in params:
                q = text
                for prefix in ["what do you know about", "knowledge about", "tell me about",
                               "know about", "knowledge base for", "kb about", "kb for",
                               "search knowledge for", "query knowledge about"]:
                    if q.lower().startswith(prefix):
                        q = q[len(prefix):].strip().lstrip(":,. ")
                        break
                params["query"] = q

            self._convert_param_types(tool, params)
            result = tool.execute(**params)

            elapsed = (time.time() - start) * 1000
            return AgentResponse(
                text=self._format_result(tool_name, result),
                tool_name=tool_name,
                tool_result=result,
                execution_time_ms=elapsed,
                auto_detected=True,
                success=result.success,
                error=result.error if not result.success else None,
            )

        return None

    def _try_exact_invocation(self, text: str) -> Optional[AgentResponse]:
        """Try exact tool invocation format: tool_name(param=value, ...) or /tool_name params"""
        # Format: /tool_name param1=value1 param2=value2
        cmd_match = re.match(r'^/(\w+)\s*(.*)', text)
        if cmd_match:
            tool_name = cmd_match.group(1).lower()
            rest = cmd_match.group(2)

            tool = self.registry.get(tool_name)
            if not tool:
                return AgentResponse(
                    text=f"Unknown tool: {tool_name}. Available: {', '.join(t.name for t in self.registry._tools.values())}",
                    success=False,
                    error=f"Unknown tool: {tool_name}",
                )

            # Parse key=value params or fall back to positional
            params = {}
            kv_pairs = re.findall(r'(\w+)=(?:([^\s"\']+)|"([^"]+)"|\'([^\']+)\')', rest)
            if kv_pairs:
                for k, v1, v2, v3 in kv_pairs:
                    val = v1 or v2 or v3
                    # Convert param types based on tool schema
                    for tp in tool.parameters:
                        if tp.name == k:
                            if tp.type == "integer":
                                val = int(val)
                            elif tp.type == "float":
                                val = float(val)
                            elif tp.type == "boolean":
                                val = val.lower() in ("true", "yes", "1")
                            break
                    params[k] = val
            elif rest.strip():
                # Try to match first required param
                for p in tool.parameters:
                    if p.required:
                        params[p.name] = rest.strip()
                        break

            self._convert_param_types(tool, params)
            result = tool.execute(**params)
            return AgentResponse(
                text=self._format_result(tool_name, result),
                tool_name=tool_name,
                tool_result=result,
                auto_detected=False,
                success=result.success,
                error=result.error if not result.success else None,
            )

        return None

    def _detect_code_block(self, text: str) -> Optional[str]:
        """Detect if text contains code (even without markdown fences)"""
        # Code block already extracted
        block = self._extract_code_block(text)
        if block:
            return block

        code_indicators = [
            r'^def\s+\w+\s*\(', r'^class\s+\w+', r'^import\s+\w+',
            r'^from\s+\w+\s+import', r'^function\s+\w+\s*\(',
            r'^const\s+\w+\s*=', r'^let\s+\w+\s*=', r'^var\s+\w+\s*=',
            r'^public\s+(?:static\s+)?\w+\s+\w+\s*\(',
            r'^fn\s+\w+\s*\(', r'^func\s+\w+\s*\(',
        ]
        lines = text.strip().split('\n')
        first_line = lines[0].strip()
        for pattern in code_indicators:
            if re.match(pattern, first_line):
                return text
        return None

    def _extract_params(self, text: str, patterns: Dict[str, str]) -> Dict[str, str]:
        """Extract parameters from text using regex patterns"""
        params = {}
        for param_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Normalize known values
                if param_name == "content_type":
                    value = value.replace(" ", "_").lower()
                params[param_name] = value
        return params

        return None

    def _convert_param_types(self, tool, params: Dict):
        """Convert string params to proper types based on tool schema"""
        for tp in tool.parameters:
            if tp.name not in params:
                continue
            val = params[tp.name]
            if tp.type == "integer":
                try:
                    params[tp.name] = int(val)
                except (ValueError, TypeError):
                    pass
            elif tp.type == "float":
                try:
                    params[tp.name] = float(val)
                except (ValueError, TypeError):
                    pass
            elif tp.type == "boolean":
                if isinstance(val, str):
                    params[tp.name] = val.lower() in ("true", "yes", "1", "on")

    def _extract_code_block(self, text: str) -> Optional[str]:
        """Extract code from markdown code blocks or inline backticks"""
        # Fenced code blocks
        block_match = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        if block_match:
            return block_match.group(1).strip()

        # Inline backticks
        inline_match = re.search(r'`([^`]+)`', text)
        if inline_match:
            return inline_match.group(1).strip()

        return None

    def _format_result(self, tool_name: str, result: ToolResult) -> str:
        """Format tool result into readable text"""
        if not result.success:
            return f"[{tool_name}] Error: {result.error}"

        if tool_name == "calculator":
            data = result.output
            return f"`{data['expression']}` = **{data['result']}**"

        if tool_name == "content_writer":
            data = result.output
            outline = data.get("outline", {})
            title = outline.get("title_suggestions", [""])[0]
            sections = outline.get("sections", [])
            return f"**Content Outline: {title}**\n\n" + \
                   f"Type: {data['content_type']} | Tone: {data['tone']} | Length: {data['length']}\n\n" + \
                   "\n".join(f"- **{s['heading']}**: {s['purpose']}" for s in sections)

        if tool_name == "summarizer":
            data = result.output
            summary = data.get("summary") or data.get("key_points") or data.get("bullets") or ""
            extra = ""
            if data.get("keywords"):
                extra = f"\n\nKeywords: {', '.join(data['keywords'])}"
            if data.get("compression_ratio"):
                extra += f"\nCompression: {data['compression_ratio']}% of original"
            return f"**Summary**\n\n{summary}{extra}"

        if tool_name == "code_review":
            data = result.output
            issues = data.get("issues", [])
            by_sev = data.get("issues_by_severity", {})
            header = f"**Code Review: {data['assessment']}**\n"
            header += f"Lines: {data['stats']['lines']} | Functions: {data['stats']['functions']} | Issues: {data['issue_count']}\n"
            by_sev_str = " | ".join(f"{k}: {v}" for k, v in by_sev.items() if v > 0)
            header += f"({by_sev_str})\n"
            if issues:
                header += "\nIssues:\n"
                for i in issues[:10]:
                    header += f"- [{i['severity']}] Line {i['line']}: {i['message']}\n"
            return header

        if tool_name == "text_analyzer":
            data = result.output
            parts = []
            if "sentiment" in data:
                s = data["sentiment"]
                parts.append(f"Sentiment: {s['label']} (score: {s['compound']})")
            if "readability" in data:
                r = data["readability"]
                parts.append(f"Readability: {r['reading_level']} (Flesch: {r['flesch_reading_ease']})")
            if "statistics" in data:
                st = data["statistics"]
                parts.append(f"Stats: {st['words']} words, {st['sentences']} sentences, {st['paragraphs']} paragraphs")
            if "keywords" in data:
                kws = [k["word"] for k in data["keywords"][:5]]
                parts.append(f"Keywords: {', '.join(kws)}")
            if "entities" in data:
                for etype, ents in data["entities"].items():
                    parts.append(f"{etype}: {', '.join(ents[:3])}")
            if "tone" in data:
                parts.append(f"Tone: {data['tone']['primary_tone']} ({data['tone']['formality']})")
            return "**Text Analysis**\n\n" + "\n".join(f"- {p}" for p in parts)

        if tool_name == "regex_builder":
            data = result.output
            if "generated_pattern" in data:
                return f"**Pattern: {data['pattern_name'] if 'pattern_name' in data else 'generated'}**\n\n`{data['generated_pattern']}`"
            if "matches" in data:
                return f"**Matches: {data.get('match_count', len(data.get('matches', [])))}**\n\n" + \
                       "\n".join(f"- `{m}`" for m in data.get("matches", [])[:10])
            return f"Pattern: `{data.get('pattern', '')}`"

        if tool_name == "json_builder":
            data = result.output
            if "valid" in data:
                if data.get("parsed") if isinstance(data, dict) else False:
                    return f"✅ Valid JSON ({data.get('type', '')})"
                return f"❌ Invalid JSON: {data.get('error', '')}"
            for key in ("formatted", "minified", "merged", "result"):
                if key in data:
                    return f"```json\n{data[key]}\n```"
            return f"```json\n{result.output}\n```"

        if tool_name == "markdown_builder":
            data = result.output
            md = data.get("markdown", "")
            return f"```markdown\n{md}\n```"

        if tool_name == "test_generator":
            data = result.output
            tc = data.get("test_code", "")
            return f"```python\n{tc}\n```" + \
                   f"\n(Functions: {data.get('functions_tested', 0)}, Classes: {data.get('classes_tested', 0)})"

        if tool_name == "color_palette":
            data = result.output
            return f"**Color Palette ({data['scheme']})**\nBase: {data['base_color']}\n" + \
                   "\n".join(f"- {c}" for c in data["colors"]) + "\n\n" + \
                   "Accessibility:\n" + \
                   "\n".join(f"- {k}: ratio {v['contrast_ratio']}, AA={v['passes_AA_normal']}"
                            for k, v in data.get("accessibility", {}).items())

        if tool_name == "password_generator":
            data = result.output
            pwds = data.get("passwords", [])
            if isinstance(pwds, dict):
                pwds = [pwds]
            return "**Generated Passwords**\n" + \
                   "\n".join(f"- `{p['password']}` (strength: {p['strength']['label']}, "
                            f"entropy: {p['strength']['entropy_bits']} bits)"
                            for p in pwds)

        if tool_name == "hash_tool":
            data = result.output
            return f"**{data['algorithm'].upper()}**: `{data['hash']}`"

        if tool_name == "text_to_sql":
            data = result.output
            return f"**SQL Query**\n\n```sql\n{data['sql']}\n```\n\n{data.get('explanation', '')}"

        if tool_name == "translator":
            data = result.output
            return f"**Translation to {data['target_language']}**\n\nOriginal: {data['original']}"

        if tool_name == "web_search":
            data = result.output
            if isinstance(data, dict):
                data = [data]
            lines = [f"**Search Results for: {result.metadata.get('query', '')}**\n"]
            for r in data[:10]:
                lines.append(f"- **{r['title']}**")
                lines.append(f"  URL: {r['url']}")
                if r.get('snippet'):
                    lines.append(f"  _{r['snippet']}_")
                lines.append("")
            return "\n".join(lines)

        if tool_name == "knowledge_query":
            data = result.output
            if isinstance(data, dict):
                data = [data]
            if not data:
                return "**Knowledge Base**: No results found."
            lines = [f"**Knowledge Results**\n"]
            for r in data[:10]:
                lines.append(f"- **{r['title']}** (score: {r['score']:.3f})")
                if r.get('url'):
                    lines.append(f"  URL: {r['url']}")
                if r.get('summary'):
                    lines.append(f"  _{r['summary'][:200]}_")
                lines.append("")
            return "\n".join(lines)

        if tool_name == "api_caller":
            data = result.output
            body = data.get("body", {})
            return f"**HTTP {data['status_code']} {data['status_text']}**\n" + \
                   (f"```json\n{body}\n```" if body else "")

        if tool_name == "simulator":
            data = result.output
            lines = [f"**{data['simulation']}**"]
            lines.append(f"Steps: {data.get('steps', 0)} | Final time: {data.get('final_time', 0):.4e}")
            if data.get("energy") is not None:
                lines.append(f"Total energy: {data['energy']:.4e}")
            if data.get("temperature") is not None:
                lines.append(f"Temperature: {data['temperature']:.2f} K")
            if data.get("stocks"):
                lines.append("")
                for s in data["stocks"]:
                    lines.append(f"- {s['name']}: {s['value']:.2f}")
            elif data.get("entities"):
                lines.append("")
                for e in data["entities"][:10]:
                    pos_str = f"pos=[{e['position'][0]:.4e}, {e['position'][1]:.4e}, {e['position'][2]:.4e}]" if "position" in e else ""
                    vel_str = f" vel=[{e['velocity'][0]:.4e}, ..]" if "velocity" in e else ""
                    val_str = f" value={e['value']:.4e}" if "value" in e else ""
                    lines.append(f"- Entity #{e['id']} ({e['type']}): {pos_str}{vel_str}{val_str}")
            return "\n".join(lines)

        # Generic fallback for any tool
        out = result.output
        if isinstance(out, dict):
            lines = []
            for k, v in out.items():
                if isinstance(v, (str, int, float, bool)):
                    lines.append(f"{k}: {v}")
                elif isinstance(v, list) and len(v) < 10:
                    lines.append(f"{k}: {', '.join(str(x) for x in v)}")
            return "\n".join(lines) if lines else str(out)
        return str(out)[:1000]

    def list_tools(self) -> str:
        """List all available tools with descriptions"""
        tools = self.registry.list_tools()
        lines = ["**Available Tools**\n"]
        for t in tools:
            lines.append(f"- **{t['name']}**: {t['description']}")
        return "\n".join(lines)
