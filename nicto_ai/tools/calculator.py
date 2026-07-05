"""
NICTO AI - Calculator Tool
Quick calculator for arithmetic and common math operations.
"""

import math
import logging
from typing import Dict
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class CalculatorTool(Tool):
    """
    Quick calculator for common mathematical operations.

    Faster than the math_engine for simple calculations.
    """

    name = "calculator"
    description = "Quick calculator for arithmetic and common math operations. Fast and simple."
    parameters = [
        ToolParameter(name="expression", type="string", description="Mathematical expression to calculate", required=True),
    ]
    tags = ["math", "calculator", "arithmetic"]
    timeout_seconds = 5.0

    SAFE_FUNCTIONS = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "cbrt": lambda x: x ** (1/3),
        "pow": pow, "log": math.log, "log2": math.log2, "log10": math.log10,
        "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "ceil": math.ceil, "floor": math.floor,
        "factorial": math.factorial, "gcd": math.gcd,
        "radians": math.radians, "degrees": math.degrees,
    }

    def _execute(self, expression: str) -> ToolResult:
        try:
            result = eval(expression, {"__builtins__": {}}, self.SAFE_FUNCTIONS)
            return ToolResult(
                success=True,
                output={
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {str(e)}")
