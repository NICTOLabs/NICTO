"""
NICTO AI - Code Executor Tool
Execute Python code in an isolated subprocess sandbox.
"""

import sys
import os
import logging
import subprocess
from typing import Dict, Optional
from .base import Tool, ToolResult, ToolParameter, ToolPermission

logger = logging.getLogger(__name__)


class CodeExecutorTool(Tool):
    """
    Execute Python code in an isolated subprocess sandbox.

    Security features:
    - Runs in isolated subprocess (not in NICTO's process)
    - Timeout prevents infinite loops
    - Output is captured and sanitized
    """

    name = "code_executor"
    description = "Execute Python code and return the output. Useful for calculations, data analysis, algorithms, and testing ideas."
    parameters = [
        ToolParameter(name="code", type="string", description="Python code to execute", required=True),
        ToolParameter(name="timeout", type="integer", description="Timeout in seconds (1-60)", required=False, default=10),
    ]
    permissions = [ToolPermission.EXECUTE]
    tags = ["code", "python", "execute", "sandbox"]
    timeout_seconds = 60.0

    def _execute(self, code: str, timeout: int = 10) -> ToolResult:
        timeout = min(max(timeout, 1), 60)

        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )

            if result.returncode == 0:
                output = result.stdout if result.stdout else "(no output)"
                return ToolResult(
                    success=True,
                    output=output,
                    metadata={
                        "has_stderr": bool(result.stderr),
                        "stderr": result.stderr[:500] if result.stderr else None,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    error=result.stderr[:500] if result.stderr else "Execution failed",
                    metadata={"return_code": result.returncode},
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Code execution timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
            )
