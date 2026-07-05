"""
NICTO AI - Base Tool Framework
Defines the abstract Tool interface and registry system.

All NICTO tools implement the Tool interface, which provides:
- Name and description for LLM function calling
- Input/output schema for validation
- Execution with timeout and error handling
- Permission checking
"""

import time
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ToolPermission(Enum):
    """Permission levels for tools"""
    READ = "read"         # Read-only access
    WRITE = "write"       # Write access to files/system
    EXECUTE = "execute"   # Execute code/commands
    NETWORK = "network"   # Access network/internet
    ADMIN = "admin"       # Full system access


@dataclass
class ToolParameter:
    """Schema for a tool parameter"""
    name: str
    type: str  # "string", "integer", "float", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None

    def to_schema(self) -> Dict:
        """Convert to JSON Schema format"""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "output": self.output if self.success else None,
            "error": self.error if not self.success else None,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }


class Tool(ABC):
    """
    Abstract base class for all NICTO tools.

    To create a new tool:
    1. Subclass Tool
    2. Implement name, description, parameters
    3. Implement _execute()
    4. Register with ToolRegistry

    Example:
        class MyTool(Tool):
            name = "my_tool"
            description = "Does something useful"
            parameters = [ToolParameter(...)]

            def _execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="done")
    """

    name: str = "base_tool"
    description: str = "Base tool"
    parameters: List[ToolParameter] = []
    permissions: List[ToolPermission] = []
    tags: List[str] = []
    timeout_seconds: float = 30.0
    enabled: bool = True

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with error handling and timing.

        Do NOT override this method. Override _execute() instead.
        """
        start_time = time.time()

        # Validate parameters
        validation_error = self._validate_params(kwargs)
        if validation_error:
            return ToolResult(
                success=False,
                error=f"Parameter validation failed: {validation_error}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            result = self._execute(**kwargs)
            result.execution_time_ms = (time.time() - start_time) * 1000
            logger.debug("Tool %s executed in %.1fms", self.name, result.execution_time_ms)
            return result
        except TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool {self.name} timed out after {self.timeout_seconds}s",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error("Tool %s failed: %s", self.name, e)
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={"traceback": traceback.format_exc()},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    @abstractmethod
    def _execute(self, **kwargs) -> ToolResult:
        """Implement the tool's actual logic here"""
        pass

    def get_schema(self) -> Dict:
        """Get the tool's schema for LLM function calling"""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _validate_params(self, kwargs: Dict) -> Optional[str]:
        """Validate input parameters against schema"""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return f"Missing required parameter: {param.name}"
            if param.name in kwargs and param.enum:
                if kwargs[param.name] not in param.enum:
                    return f"Parameter {param.name} must be one of: {param.enum}"
        return None

    def __repr__(self):
        return f"Tool({self.name}: {self.description[:50]})"


class ToolRegistry:
    """
    Registry for all NICTO tools.

    Manages tool registration, lookup, and execution.
    Tools are registered by name and can be looked up
    for execution or schema generation.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: Tool, category: str = "general"):
        """Register a tool"""
        self._tools[tool.name] = tool
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)
        logger.debug("Registered tool: %s (category: %s)", tool.name, category)

    def unregister(self, name: str):
        """Unregister a tool"""
        if name in self._tools:
            del self._tools[name]
            for cat, tools in self._categories.items():
                if name in tools:
                    tools.remove(name)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool not found: {name}",
            )
        if not tool.enabled:
            return ToolResult(
                success=False,
                error=f"Tool disabled: {name}",
            )
        return tool.execute(**kwargs)

    def list_tools(self, category: Optional[str] = None) -> List[Dict]:
        """List all tools, optionally filtered by category"""
        if category:
            tool_names = self._categories.get(category, [])
            return [self._tools[n].get_schema() for n in tool_names if n in self._tools]
        return [t.get_schema() for t in self._tools.values()]

    def get_schemas_for_llm(self) -> List[Dict]:
        """Get all tool schemas formatted for LLM function calling"""
        return [t.get_schema() for t in self._tools.values() if t.enabled]

    def list_categories(self) -> List[str]:
        """List all tool categories"""
        return list(self._categories.keys())

    def enable_tool(self, name: str):
        """Enable a tool"""
        if name in self._tools:
            self._tools[name].enabled = True

    def disable_tool(self, name: str):
        """Disable a tool"""
        if name in self._tools:
            self._tools[name].enabled = False

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"ToolRegistry({len(self._tools)} tools)"
