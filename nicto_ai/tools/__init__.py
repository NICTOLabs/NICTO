"""NICTO AI Tool System - Extensible tool ecosystem for real-world capabilities"""
from .base import Tool, ToolResult, ToolRegistry, ToolParameter, ToolPermission
from .web_search import WebSearchTool
from .code_executor import CodeExecutorTool
from .file_manager import FileManagerTool
from .math_engine import MathEngineTool
from .data_analysis import DataAnalysisTool
from .translator import TranslatorTool
from .knowledge_tool import KnowledgeTool
from .simulation_tool import SimulationTool
from .calculator import CalculatorTool
from .shell import ShellTool
from .content_writer import ContentWriterTool
from .summarizer import SummarizerTool
from .code_review import CodeReviewTool
from .text_analyzer import TextAnalyzerTool
from .regex_builder import RegexBuilderTool
from .json_builder import JsonBuilderTool
from .markdown_builder import MarkdownBuilderTool
from .test_generator import UnitTestGeneratorTool
from .api_caller import ApiCallerTool
from .password_generator import PasswordGeneratorTool
from .hash_tool import HashTool
from .color_palette import ColorPaletteTool
from .text_to_sql import TextToSQLTool

__all__ = [
    "Tool", "ToolResult", "ToolRegistry", "ToolParameter", "ToolPermission",
    "WebSearchTool", "CodeExecutorTool", "FileManagerTool",
    "MathEngineTool", "DataAnalysisTool", "TranslatorTool",
    "KnowledgeTool", "CalculatorTool", "ShellTool",
    "ContentWriterTool", "SummarizerTool", "CodeReviewTool",
    "TextAnalyzerTool", "RegexBuilderTool", "JsonBuilderTool",
    "MarkdownBuilderTool", "UnitTestGeneratorTool", "ApiCallerTool",
    "PasswordGeneratorTool", "HashTool",     "ColorPaletteTool",
    "TextToSQLTool",
    "SimulationTool",
]


def create_default_registry(browser=None, knowledge_base=None) -> ToolRegistry:
    """Create and populate the default tool registry"""
    from .content_writer import ContentWriterTool
    from .summarizer import SummarizerTool
    from .code_review import CodeReviewTool
    from .text_analyzer import TextAnalyzerTool
    from .regex_builder import RegexBuilderTool
    from .json_builder import JsonBuilderTool
    from .markdown_builder import MarkdownBuilderTool
    from .test_generator import UnitTestGeneratorTool
    from .api_caller import ApiCallerTool
    from .password_generator import PasswordGeneratorTool
    from .hash_tool import HashTool
    from .color_palette import ColorPaletteTool
    from .text_to_sql import TextToSQLTool

    registry = ToolRegistry()

    # Core tools
    registry.register(CodeExecutorTool(), category="development")
    registry.register(ShellTool(), category="development")
    registry.register(CalculatorTool(), category="math")
    registry.register(MathEngineTool(), category="math")
    registry.register(FileManagerTool(), category="file")

    # Knowledge tools
    registry.register(WebSearchTool(), category="knowledge")
    registry.register(KnowledgeTool(knowledge_base=knowledge_base), category="knowledge")

    # Data tools
    registry.register(DataAnalysisTool(), category="data")
    registry.register(JsonBuilderTool(), category="data")
    registry.register(TextToSQLTool(), category="data")

    # Content tools
    registry.register(ContentWriterTool(), category="content")
    registry.register(SummarizerTool(), category="content")
    registry.register(MarkdownBuilderTool(), category="content")

    # Analysis tools
    registry.register(TextAnalyzerTool(), category="analysis")
    registry.register(CodeReviewTool(), category="analysis")
    registry.register(RegexBuilderTool(), category="analysis")

    # Utility tools
    registry.register(TranslatorTool(), category="utility")
    registry.register(ApiCallerTool(), category="utility")
    registry.register(PasswordGeneratorTool(), category="utility")
    registry.register(HashTool(), category="utility")
    registry.register(ColorPaletteTool(), category="utility")

    # Testing tools
    registry.register(UnitTestGeneratorTool(), category="testing")

    # Simulation tool
    registry.register(SimulationTool(), category="simulation")

    return registry

