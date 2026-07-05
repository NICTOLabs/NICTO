"""
Test NICTO AI Tool System
Tool registry, all tool implementations
"""

import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nicto_ai.tools.base import Tool, ToolResult, ToolRegistry, ToolParameter
from nicto_ai.tools.calculator import CalculatorTool
from nicto_ai.tools.code_executor import CodeExecutorTool
from nicto_ai.tools.file_manager import FileManagerTool
from nicto_ai.tools.math_engine import MathEngineTool
from nicto_ai.tools.data_analysis import DataAnalysisTool
from nicto_ai.tools.shell import ShellTool


# ─── ToolRegistry Tests ─────────────────────────────────────────

def test_registry_init():
    reg = ToolRegistry()
    assert len(reg) == 0
    print("  Registry init: OK")

def test_registry_register():
    reg = ToolRegistry()
    tool = CalculatorTool()
    reg.register(tool, category="math")
    assert len(reg) == 1
    assert reg.get("calculator") is not None
    print("  Registry register: OK")

def test_registry_execute():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    result = reg.execute("calculator", expression="2 + 2")
    assert result.success
    assert result.output["result"] == 4
    print("  Registry execute: OK")

def test_registry_list():
    reg = ToolRegistry()
    reg.register(CalculatorTool(), "math")
    reg.register(ShellTool(), "system")
    tools = reg.list_tools()
    assert len(tools) == 2
    print("  Registry list: OK")

def test_registry_category():
    reg = ToolRegistry()
    reg.register(CalculatorTool(), "math")
    tools = reg.list_tools(category="math")
    assert len(tools) == 1
    print("  Registry category: OK")

def test_registry_schemas():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    schemas = reg.get_schemas_for_llm()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "calculator"
    print("  Registry schemas: OK")

def test_registry_disable():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.disable_tool("calculator")
    result = reg.execute("calculator", expression="1+1")
    assert not result.success
    print("  Registry disable: OK")


# ─── CalculatorTool Tests ───────────────────────────────────────

def test_calculator_basic():
    tool = CalculatorTool()
    result = tool.execute(expression="2 + 2")
    assert result.success
    assert result.output["result"] == 4
    print("  Calculator basic: OK")

def test_calculator_complex():
    tool = CalculatorTool()
    result = tool.execute(expression="sqrt(16) + pow(2, 3)")
    assert result.success
    assert result.output["result"] == 12.0
    print("  Calculator complex: OK")

def test_calculator_trig():
    tool = CalculatorTool()
    result = tool.execute(expression="sin(pi/2)")
    assert result.success
    assert abs(result.output["result"] - 1.0) < 1e-10
    print("  Calculator trig: OK")

def test_calculator_error():
    tool = CalculatorTool()
    result = tool.execute(expression="1/0")
    assert not result.success
    print("  Calculator error: OK")

def test_calculator_schema():
    tool = CalculatorTool()
    schema = tool.get_schema()
    assert "name" in schema
    assert "parameters" in schema
    print("  Calculator schema: OK")


# ─── CodeExecutorTool Tests ─────────────────────────────────────

def test_executor_basic():
    tool = CodeExecutorTool()
    result = tool.execute(code="print(2 + 2)")
    assert result.success
    assert "4" in result.output
    print("  Executor basic: OK")

def test_executor_output_capture():
    tool = CodeExecutorTool()
    result = tool.execute(code="print('hello'); print('world')")
    assert result.success
    assert "hello" in result.output
    assert "world" in result.output
    print("  Executor output capture: OK")

def test_executor_error():
    tool = CodeExecutorTool()
    result = tool.execute(code="1/0")
    assert not result.success
    print("  Executor error: OK")

def test_executor_imports():
    tool = CodeExecutorTool()
    result = tool.execute(code="import math; print(math.sqrt(16))")
    assert result.success
    assert "4.0" in result.output
    print("  Executor imports: OK")

def test_executor_timeout():
    tool = CodeExecutorTool()
    result = tool.execute(code="import time; time.sleep(60)", timeout=1)
    assert not result.success
    print("  Executor timeout: OK")


# ─── FileManagerTool Tests ──────────────────────────────────────

def test_filemanager_write_read():
    tool = FileManagerTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        result = tool.execute(operation="write", path=path, content="Hello NICTO")
        assert result.success

        result = tool.execute(operation="read", path=path)
        assert result.success
        assert result.output == "Hello NICTO"
    print("  FileManager write/read: OK")

def test_filemanager_list():
    tool = FileManagerTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.txt"), "w").close()
        open(os.path.join(tmpdir, "b.txt"), "w").close()
        result = tool.execute(operation="list", path=tmpdir)
        assert result.success
        assert len(result.output) == 2
    print("  FileManager list: OK")

def test_filemanager_exists():
    tool = FileManagerTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        result = tool.execute(operation="exists", path=path)
        assert result.success
        assert result.output == False
    print("  FileManager exists: OK")

def test_filemanager_json():
    tool = FileManagerTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json")
        data = {"key": "value", "number": 42}
        result = tool.execute(operation="write", path=path, content=json.dumps(data))
        assert result.success

        result = tool.execute(operation="read", path=path)
        assert result.success
        assert result.output["key"] == "value"
    print("  FileManager JSON: OK")


# ─── MathEngineTool Tests ───────────────────────────────────────

def test_math_evaluate():
    tool = MathEngineTool()
    result = tool.execute(expression="2 + 2", operation="evaluate")
    assert result.success
    assert result.output["result"] == 4
    print("  Math evaluate: OK")

def test_math_statistics():
    tool = MathEngineTool()
    result = tool.execute(expression="[1, 2, 3, 4, 5]", operation="statistics")
    assert result.success
    assert result.output["mean"] == 3.0
    assert result.output["count"] == 5
    print("  Math statistics: OK")

def test_math_derivative():
    tool = MathEngineTool()
    result = tool.execute(expression="x**2", operation="derivative", variable="x")
    assert result.success
    assert "derivative_approx" in result.output
    print("  Math derivative: OK")


# ─── DataAnalysisTool Tests ─────────────────────────────────────

def test_data_csv_summary():
    tool = DataAnalysisTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with open(path, "w") as f:
            f.write("name,age,score\nAlice,25,90\nBob,30,85\nCharlie,35,95\n")
        result = tool.execute(file_path=path, operation="summary")
        assert result.success
        assert result.output["total_rows"] == 3
    print("  Data CSV summary: OK")

def test_data_quality():
    tool = DataAnalysisTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with open(path, "w") as f:
            f.write("name,age\nAlice,25\nBob,\nCharlie,35\n")
        result = tool.execute(file_path=path, operation="quality")
        assert result.success
    print("  Data quality: OK")


# ─── ShellTool Tests ────────────────────────────────────────────

def test_shell_basic():
    tool = ShellTool()
    result = tool.execute(command="echo hello")
    assert result.success
    assert "hello" in result.output
    print("  Shell basic: OK")

def test_shell_error():
    tool = ShellTool()
    result = tool.execute(command="exit 1")
    assert not result.success
    print("  Shell error: OK")


# ─── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NICTO AI - Tool System Test Suite")
    print("=" * 60)

    print("\n--- ToolRegistry ---")
    test_registry_init()
    test_registry_register()
    test_registry_execute()
    test_registry_list()
    test_registry_category()
    test_registry_schemas()
    test_registry_disable()

    print("\n--- CalculatorTool ---")
    test_calculator_basic()
    test_calculator_complex()
    test_calculator_trig()
    test_calculator_error()
    test_calculator_schema()

    print("\n--- CodeExecutorTool ---")
    test_executor_basic()
    test_executor_output_capture()
    test_executor_error()
    test_executor_imports()
    test_executor_timeout()

    print("\n--- FileManagerTool ---")
    test_filemanager_write_read()
    test_filemanager_list()
    test_filemanager_exists()
    test_filemanager_json()

    print("\n--- MathEngineTool ---")
    test_math_evaluate()
    test_math_statistics()
    test_math_derivative()

    print("\n--- DataAnalysisTool ---")
    test_data_csv_summary()
    test_data_quality()

    print("\n--- ShellTool ---")
    test_shell_basic()
    test_shell_error()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
