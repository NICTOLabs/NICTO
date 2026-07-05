"""NICTO AI Tool System - Extensible tool ecosystem for real-world capabilities"""
from .base import Tool, ToolResult, ToolRegistry
from .web_search import WebSearchTool
from .code_executor import CodeExecutorTool
from .file_manager import FileManagerTool
from .math_engine import MathEngineTool
from .data_analysis import DataAnalysisTool
from .translator import TranslatorTool
from .knowledge_tool import KnowledgeTool
from .calculator import CalculatorTool
from .shell import ShellTool

__all__ = [
    "Tool", "ToolResult", "ToolRegistry",
    "WebSearchTool", "CodeExecutorTool", "FileManagerTool",
    "MathEngineTool", "DataAnalysisTool", "TranslatorTool",
    "KnowledgeTool", "CalculatorTool", "ShellTool",
]
