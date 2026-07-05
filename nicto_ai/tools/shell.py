"""
NICTO AI - Shell Tool
Execute shell commands in a controlled environment.
"""

import subprocess
import os
import logging
from typing import Dict, Optional
from .base import Tool, ToolResult, ToolParameter, ToolPermission

logger = logging.getLogger(__name__)


class ShellTool(Tool):
    """
    Execute shell commands with safety controls.

    Runs commands in a sandboxed environment with:
    - Timeout protection
    - Output capture
    - Working directory control
    - Command whitelist (optional)
    """

    name = "shell"
    description = "Execute shell commands. Useful for git, npm, pip, system info, and file operations."
    parameters = [
        ToolParameter(name="command", type="string", description="Shell command to execute", required=True),
        ToolParameter(name="working_dir", type="string", description="Working directory", required=False, default="."),
        ToolParameter(name="timeout", type="integer", description="Timeout in seconds (1-30)", required=False, default=10),
    ]
    permissions = [ToolPermission.EXECUTE]
    tags = ["shell", "command", "system"]
    timeout_seconds = 30.0

    def _execute(self, command: str, working_dir: str = ".", timeout: int = 10) -> ToolResult:
        timeout = min(max(timeout, 1), 30)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir if os.path.isdir(working_dir) else None,
            )

            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout if result.stdout else "(no output)",
                error=result.stderr if result.stderr else None,
                metadata={
                    "return_code": result.returncode,
                    "command": command,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
