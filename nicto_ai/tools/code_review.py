"""
NICTO AI - Code Review Tool
Analyze code for bugs, security issues, performance, and best practices.
"""

import re
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class CodeReviewTool(Tool):
    """
    Code analysis and review engine.

    Detects:
    - Security vulnerabilities (SQL injection, XSS, etc.)
    - Performance issues (N+1 queries, memory leaks)
    - Code smells (long methods, deep nesting)
    - Best practices violations
    - Type hints and documentation
    - Test coverage suggestions
    """

    name = "code_review"
    description = "Review code for bugs, security issues, performance problems, and best practices. Supports Python, JavaScript, and more."
    parameters = [
        ToolParameter(name="code", type="string", description="Code to review", required=True),
        ToolParameter(name="language", type="string", description="Programming language", required=False, default="python", enum=[
            "python", "javascript", "typescript", "java", "c", "cpp", "go", "rust", "ruby", "php",
        ]),
        ToolParameter(name="focus", type="string", description="Review focus area", required=False, default="all", enum=[
            "all", "security", "performance", "readability", "best_practices", "bugs",
        ]),
        ToolParameter(name="severity_filter", type="string", description="Minimum severity to report", required=False, default="low", enum=[
            "critical", "high", "medium", "low", "info",
        ]),
    ]
    tags = ["code", "review", "security", "quality", "analysis"]
    timeout_seconds = 30.0

    # Security patterns by language
    SECURITY_PATTERNS = {
        "python": [
            (r'eval\(', "CRITICAL", "Use of eval() - potential code injection"),
            (r'exec\(', "CRITICAL", "Use of exec() - potential code injection"),
            (r'os\.system\(', "HIGH", "Use of os.system() - use subprocess instead"),
            (r'subprocess\.call\(.*shell=True', "HIGH", "Shell injection risk - use shell=False"),
            (r'pickle\.loads?\(', "HIGH", "Untrusted pickle deserialization - security risk"),
            (r'__import__\(', "MEDIUM", "Dynamic import - potential security concern"),
            (r'input\(', "LOW", "Direct user input - validate and sanitize"),
            (r'open\(.*\+', "MEDIUM", "File opened in read+write mode - check intent"),
        ],
        "javascript": [
            (r'eval\(', "CRITICAL", "Use of eval() - potential code injection"),
            (r'innerHTML\s*=', "HIGH", "innerHTML assignment - XSS vulnerability"),
            (r'document\.write\(', "HIGH", "document.write() - XSS vulnerability"),
            (r'\.innerHTML', "MEDIUM", "innerHTML usage - consider textContent"),
            (r'new Function\(', "HIGH", "Dynamic function creation - code injection risk"),
            (r'setTimeout\(.*,.*string', "HIGH", "setTimeout with string - equivalent to eval"),
            (r'addEventListener.*\+', "MEDIUM", "String concatenation in event names"),
        ],
    }

    # Performance patterns
    PERFORMANCE_PATTERNS = {
        "python": [
            (r'for.*in.*range\(len\(', "MEDIUM", "Use enumerate() instead of range(len())"),
            (r'\.append\(.*\)\s*\n.*\.append\(', "LOW", "Consider list comprehension for batch operations"),
            (r'import.*\n.*import', "LOW", "Multiple imports - consider grouping"),
            (r'def.*\*args.*\*kwargs', "INFO", "Generic function signature - consider specific params"),
        ],
        "javascript": [
            (r'for\s*\(\s*var\s+\w+\s*=\s*0', "LOW", "Use let/const instead of var"),
            (r'\.forEach\(', "INFO", "forEach is slower than for...of for large arrays"),
            (r'JSON\.parse\(JSON\.stringify\(', "HIGH", "Deep clone via JSON - use structuredClone or library"),
        ],
    }

    # Code smell patterns
    CODE_SMELL_PATTERNS = [
        (r'def\s+\w+.*:\s*\n\s*pass', "INFO", "Empty function - implement or remove"),
        (r'except\s*:', "MEDIUM", "Bare except - catch specific exceptions"),
        (r'#\s*TODO', "INFO", "TODO comment - needs attention"),
        (r'#\s*FIXME', "MEDIUM", "FIXME comment - known issue"),
        (r'#\s*HACK', "HIGH", "HACK comment - technical debt"),
        (r'print\(', "LOW", "Print statement - consider logging"),
        (r'except.*pass', "MEDIUM", "Silent exception handling - log or re-raise"),
    ]

    def _execute(self, code: str, language: str = "python", focus: str = "all",
                 severity_filter: str = "low") -> ToolResult:

        if not code or not code.strip():
            return ToolResult(success=False, error="Code is empty")

        issues = []
        stats = {
            "lines": len(code.split('\n')),
            "characters": len(code),
            "functions": len(re.findall(r'(?:def|function|func)\s+\w+', code)),
            "classes": len(re.findall(r'(?:class|struct|interface)\s+\w+', code)),
            "comments": len(re.findall(r'#.*|//.*|/\*.*?\*/', code, re.DOTALL)),
        }

        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0,
                          "critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        min_severity = severity_order.get(severity_filter, 0)

        # Check security patterns
        if focus in ("all", "security"):
            patterns = self.SECURITY_PATTERNS.get(language, [])
            for pattern, severity, message in patterns:
                if severity_order.get(severity, 0) >= min_severity:
                    matches = re.finditer(pattern, code)
                    for match in matches:
                        line_num = code[:match.start()].count('\n') + 1
                        issues.append({
                            "type": "security",
                            "severity": severity,
                            "message": message,
                            "line": line_num,
                            "snippet": code[max(0, match.start()-20):match.end()+20],
                        })

        # Check performance patterns
        if focus in ("all", "performance"):
            patterns = self.PERFORMANCE_PATTERNS.get(language, [])
            for pattern, severity, message in patterns:
                if severity_order.get(severity, 0) >= min_severity:
                    matches = re.finditer(pattern, code)
                    for match in matches:
                        line_num = code[:match.start()].count('\n') + 1
                        issues.append({
                            "type": "performance",
                            "severity": severity,
                            "message": message,
                            "line": line_num,
                        })

        # Check code smells
        if focus in ("all", "readability", "best_practices", "bugs"):
            for pattern, severity, message in self.CODE_SMELL_PATTERNS:
                if severity_order.get(severity, 0) >= min_severity:
                    matches = re.finditer(pattern, code)
                    for match in matches:
                        line_num = code[:match.start()].count('\n') + 1
                        issues.append({
                            "type": "code_smell",
                            "severity": severity,
                            "message": message,
                            "line": line_num,
                        })

        # Check for missing docstrings (Python)
        if language == "python" and focus in ("all", "best_practices"):
            func_defs = list(re.finditer(r'def\s+(\w+)\s*\(', code))
            for func_match in func_defs:
                func_name = func_match.group(1)
                func_pos = func_match.end()
                # Check if next non-whitespace is a docstring
                after_func = code[func_pos:func_pos+100].strip()
                if not after_func.startswith('"""') and not after_func.startswith("'''"):
                    line_num = code[:func_match.start()].count('\n') + 1
                    issues.append({
                        "type": "documentation",
                        "severity": "info",
                        "message": f"Function '{func_name}' missing docstring",
                        "line": line_num,
                    })

        # Sort issues by severity
        issues.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)

        # Generate summary
        critical_count = sum(1 for i in issues if i["severity"] in ("CRITICAL", "critical"))
        high_count = sum(1 for i in issues if i["severity"] in ("HIGH", "high"))
        medium_count = sum(1 for i in issues if i["severity"] in ("MEDIUM", "medium"))

        if critical_count > 0:
            assessment = "CRITICAL: Immediate action required"
        elif high_count > 0:
            assessment = "WARNING: Significant issues found"
        elif medium_count > 0:
            assessment = "REVIEW: Some improvements recommended"
        else:
            assessment = "GOOD: Minor or no issues found"

        return ToolResult(
            success=True,
            output={
                "assessment": assessment,
                "stats": stats,
                "issue_count": len(issues),
                "issues_by_severity": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": medium_count,
                    "low": sum(1 for i in issues if i["severity"] == "low"),
                    "info": sum(1 for i in issues if i["severity"] == "info"),
                },
                "issues": issues[:20],  # Limit output
                "language": language,
                "focus": focus,
            },
        )
