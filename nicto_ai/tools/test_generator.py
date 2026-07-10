"""
NICTO AI - Unit Test Generator Tool
Generate unit tests for Python and JavaScript code.
"""

import re
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class UnitTestGeneratorTool(Tool):
    """
    Unit test generation engine.

    Features:
    - Generate pytest tests for Python
    - Generate Jest tests for JavaScript
    - Test edge cases
    - Mock external dependencies
    - Assert various types
    """

    name = "test_generator"
    description = "Generate unit tests for Python (pytest) or JavaScript (Jest) code."
    parameters = [
        ToolParameter(name="code", type="string", description="Code to generate tests for", required=True),
        ToolParameter(name="language", type="string", description="Programming language", required=False, default="python", enum=[
            "python", "javascript",
        ]),
        ToolParameter(name="framework", type="string", description="Test framework", required=False, default=None),
        ToolParameter(name="test_type", type="string", description="Type of tests to generate", required=False, default="unit", enum=[
            "unit", "edge_cases", "integration", "all",
        ]),
    ]
    tags = ["testing", "pytest", "jest", "unit_tests", "tdd"]
    timeout_seconds = 30.0

    def _execute(self, code: str, language: str = "python", framework: str = None,
                 test_type: str = "unit") -> ToolResult:

        if not code or not code.strip():
            return ToolResult(success=False, error="Code is empty")

        if language == "python":
            return self._generate_python_tests(code, framework, test_type)
        elif language == "javascript":
            return self._generate_js_tests(code, framework, test_type)
        else:
            return ToolResult(success=False, error=f"Unsupported language: {language}")

    def _generate_python_tests(self, code: str, framework: str = None, test_type: str = "unit") -> ToolResult:
        """Generate pytest tests for Python code"""
        framework = framework or "pytest"

        # Extract functions and classes
        functions = re.findall(r'def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*\w+)?:', code)
        classes = re.findall(r'class\s+(\w+)(?:\(([^)]*)\))?:', code)

        test_code = '"""Auto-generated unit tests"""\n\n'
        test_code += 'import pytest\n\n'

        # Generate tests for functions
        for func_name, params in functions:
            if func_name.startswith('_'):
                continue  # Skip private functions

            param_list = [p.strip().split(':')[0].strip() for p in params.split(',') if p.strip() and p.strip() != 'self']
            param_list = [p for p in param_list if p]

            test_code += f'\nclass Test{func_name.title()}:\n'
            test_code += f'    """Tests for {func_name}"""\n\n'

            # Happy path test
            test_code += f'    def test_{func_name}_happy_path(self):\n'
            test_code += f'        """Test {func_name} with valid inputs"""\n'
            if param_list:
                args = ', '.join(['"test_value"' if 'str' in p else '42' for p in param_list])
                test_code += f'        result = {func_name}({args})\n'
            else:
                test_code += f'        result = {func_name}()\n'
            test_code += f'        assert result is not None\n\n'

            # Edge case tests
            if test_type in ("edge_cases", "all"):
                test_code += f'    def test_{func_name}_empty_input(self):\n'
                test_code += f'        """Test {func_name} with empty inputs"""\n'
                if param_list:
                    args = ', '.join(['""' if 'str' in p else '0' for p in param_list])
                    test_code += f'        result = {func_name}({args})\n'
                    test_code += f'        assert result is not None  # Handle empty gracefully\n\n'

                test_code += f'    def test_{func_name}_none_input(self):\n'
                test_code += f'        """Test {func_name} with None inputs"""\n'
                if param_list:
                    args = ', '.join(['None' for _ in param_list])
                    test_code += f'        with pytest.raises((TypeError, ValueError)):\n'
                    test_code += f'            {func_name}({args})\n\n'

        # Generate tests for classes
        for class_name, bases in classes:
            if class_name.startswith('_'):
                continue

            test_code += f'\nclass Test{class_name}:\n'
            test_code += f'    """Tests for {class_name}"""\n\n'

            test_code += f'    def test_init(self):\n'
            test_code += f'        """Test {class_name} initialization"""\n'
            test_code += f'        instance = {class_name}()\n'
            test_code += f'        assert instance is not None\n\n'

            # Find methods
            methods = re.findall(rf'def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*\w+)?:', code)
            for method in methods:
                if method.startswith('_') or method == '__init__':
                    continue
                test_code += f'    def test_{class_name.lower()}_{method}(self):\n'
                test_code += f'        """Test {class_name}.{method}"""\n'
                test_code += f'        instance = {class_name}()\n'
                test_code += f'        result = instance.{method}()\n'
                test_code += f'        assert result is not None\n\n'

        # Add fixtures
        if test_type in ("integration", "all"):
            test_code += '\n# Fixtures\n'
            test_code += '@pytest.fixture\ndef sample_data():\n'
            test_code += '    """Provide sample data for tests"""\n'
            test_code += '    return {"key": "value", "number": 42}\n\n'

        return ToolResult(
            success=True,
            output={
                "language": "python",
                "framework": framework,
                "functions_tested": len(functions),
                "classes_tested": len(classes),
                "test_code": test_code,
            },
        )

    def _generate_js_tests(self, code: str, framework: str = None, test_type: str = "unit") -> ToolResult:
        """Generate Jest tests for JavaScript code"""
        framework = framework or "jest"

        # Extract functions and classes
        functions = re.findall(r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s*)?\([^)]*\)|\([^)]*\))', code)
        classes = re.findall(r'class\s+(\w+)', code)

        test_code = f'// Auto-generated {framework} tests\n\n'

        # Generate tests for functions
        for func_name in functions:
            if func_name.startswith('_'):
                continue

            test_code += f'describe(\'{func_name}\', () => {{\n'
            test_code += f'  test(\'should work with valid input\', () => {{\n'
            test_code += f'    const result = {func_name}();\n'
            test_code += f'    expect(result).toBeDefined();\n'
            test_code += f'  }});\n\n'

            if test_type in ("edge_cases", "all"):
                test_code += f'  test(\'should handle empty input\', () => {{\n'
                test_code += f'    const result = {func_name}();\n'
                test_code += f'    expect(result).toBeDefined();\n'
                test_code += f'  }});\n\n'

            test_code += f'  test(\'should handle invalid input\', () => {{\n'
            test_code += f'    expect(() => {func_name}(null)).toThrow();\n'
            test_code += f'  }});\n'
            test_code += f'}});\n\n'

        # Generate tests for classes
        for class_name in classes:
            test_code += f'describe(\'{class_name}\', () => {{\n'
            test_code += f'  test(\'should create instance\', () => {{\n'
            test_code += f'    const instance = new {class_name}();\n'
            test_code += f'    expect(instance).toBeDefined();\n'
            test_code += f'  }});\n'
            test_code += f'}});\n\n'

        return ToolResult(
            success=True,
            output={
                "language": "javascript",
                "framework": framework,
                "functions_tested": len(functions),
                "classes_tested": len(classes),
                "test_code": test_code,
            },
        )
