"""
NICTO AI - File Manager Tool
Read, write, and manage files.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter, ToolPermission

logger = logging.getLogger(__name__)


class FileManagerTool(Tool):
    """
    File management operations.

    Supports: read, write, list, exists, delete, copy, move
    for common file types (txt, json, csv, md, py, etc.)
    """

    name = "file_manager"
    description = "Read, write, and manage files. Supports text, JSON, CSV, Markdown, and code files."
    parameters = [
        ToolParameter(name="operation", type="string", description="Operation to perform", required=True, enum=["read", "write", "list", "exists", "info"]),
        ToolParameter(name="path", type="string", description="File or directory path", required=True),
        ToolParameter(name="content", type="string", description="Content to write (for write operation)", required=False),
        ToolParameter(name="encoding", type="string", description="File encoding", required=False, default="utf-8"),
    ]
    permissions = [ToolPermission.READ, ToolPermission.WRITE]
    tags = ["file", "read", "write", "manage"]
    timeout_seconds = 10.0

    # Allowed directories (sandbox)
    _allowed_dirs: Optional[List[str]] = None

    def set_allowed_dirs(self, dirs: List[str]):
        """Set allowed directories for file operations"""
        self._allowed_dirs = [os.path.abspath(d) for d in dirs]

    def _execute(self, operation: str, path: str, content: str = None, encoding: str = "utf-8") -> ToolResult:
        path = os.path.abspath(path)

        # Security check
        if self._allowed_dirs:
            allowed = any(path.startswith(d) for d in self._allowed_dirs)
            if not allowed:
                return ToolResult(
                    success=False,
                    error=f"Access denied: path {path} is outside allowed directories",
                )

        try:
            if operation == "read":
                return self._read_file(path, encoding)
            elif operation == "write":
                return self._write_file(path, content, encoding)
            elif operation == "list":
                return self._list_directory(path)
            elif operation == "exists":
                return ToolResult(success=True, output=os.path.exists(path))
            elif operation == "info":
                return self._file_info(path)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _read_file(self, path: str, encoding: str) -> ToolResult:
        if not os.path.exists(path):
            return ToolResult(success=False, error=f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        with open(path, "r", encoding=encoding) as f:
            if ext == ".json":
                data = json.load(f)
                return ToolResult(success=True, output=data, metadata={"type": "json"})
            elif ext == ".csv":
                reader = csv.reader(f)
                rows = list(reader)
                return ToolResult(success=True, output=rows, metadata={"type": "csv", "rows": len(rows)})
            else:
                content = f.read()
                return ToolResult(
                    success=True,
                    output=content,
                    metadata={"type": "text", "size_bytes": len(content.encode(encoding))},
                )

    def _write_file(self, path: str, content: str, encoding: str) -> ToolResult:
        if content is None:
            return ToolResult(success=False, error="Content is required for write operation")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        ext = os.path.splitext(path)[1].lower()

        with open(path, "w", encoding=encoding) as f:
            if ext == ".json":
                json.dump(json.loads(content) if isinstance(content, str) else content, f, indent=2)
            else:
                f.write(content)

        return ToolResult(
            success=True,
            output=f"File written: {path}",
            metadata={"size_bytes": os.path.getsize(path)},
        )

    def _list_directory(self, path: str) -> ToolResult:
        if not os.path.exists(path):
            return ToolResult(success=False, error=f"Directory not found: {path}")
        if not os.path.isdir(path):
            return ToolResult(success=False, error=f"Not a directory: {path}")

        entries = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            entries.append({
                "name": entry,
                "type": "directory" if os.path.isdir(full_path) else "file",
                "size": os.path.getsize(full_path) if os.path.isfile(full_path) else 0,
            })

        return ToolResult(success=True, output=entries, metadata={"count": len(entries)})

    def _file_info(self, path: str) -> ToolResult:
        if not os.path.exists(path):
            return ToolResult(success=False, error=f"Path not found: {path}")

        stat = os.stat(path)
        return ToolResult(
            success=True,
            output={
                "path": path,
                "name": os.path.basename(path),
                "type": "directory" if os.path.isdir(path) else "file",
                "size_bytes": stat.st_size,
                "extension": os.path.splitext(path)[1],
            },
        )
