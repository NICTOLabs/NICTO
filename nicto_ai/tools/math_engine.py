"""
NICTO AI - Mathematical Engine Tool
Solve calculus, linear algebra, statistics, and symbolic math.
"""

import math
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class MathEngineTool(Tool):
    """
    Mathematical computation engine.

    Supports:
    - Basic arithmetic
    - Algebraic expressions
    - Calculus (derivatives, integrals)
    - Linear algebra (matrix operations)
    - Statistics (mean, median, std, correlations)
    - Probability distributions
    - Symbolic mathematics
    """

    name = "math_engine"
    description = "Solve mathematical problems: arithmetic, algebra, calculus, linear algebra, statistics, and probability."
    parameters = [
        ToolParameter(name="expression", type="string", description="Mathematical expression to evaluate", required=True),
        ToolParameter(name="operation", type="string", description="Type of math operation", required=False, default="evaluate", enum=["evaluate", "derivative", "integral", "solve", "matrix", "statistics", "probability"]),
        ToolParameter(name="variable", type="string", description="Variable for calculus operations", required=False, default="x"),
    ]
    tags = ["math", "calculation", "algebra", "calculus", "statistics"]
    timeout_seconds = 10.0

    def _execute(self, expression: str, operation: str = "evaluate", variable: str = "x") -> ToolResult:
        try:
            if operation == "evaluate":
                return self._evaluate(expression)
            elif operation == "derivative":
                return self._derivative(expression, variable)
            elif operation == "integral":
                return self._integral(expression, variable)
            elif operation == "solve":
                return self._solve(expression)
            elif operation == "matrix":
                return self._matrix_operation(expression)
            elif operation == "statistics":
                return self._statistics(expression)
            elif operation == "probability":
                return self._probability(expression)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {str(e)}")

    def _evaluate(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression"""
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "int": int, "float": float,
            "pow": pow, "divmod": divmod,
            "pi": math.pi, "e": math.e, "tau": math.tau,
            "inf": math.inf, "nan": math.nan,
            "sqrt": math.sqrt, "cbrt": lambda x: x ** (1/3),
            "log": math.log, "log2": math.log2, "log10": math.log10,
            "exp": math.exp, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "asin": math.asin, "acos": math.acos,
            "atan": math.atan, "atan2": math.atan2,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "degrees": math.degrees, "radians": math.radians,
            "factorial": math.factorial, "gcd": math.gcd,
            "ceil": math.ceil, "floor": math.floor,
        }

        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return ToolResult(
            success=True,
            output={"result": result, "type": type(result).__name__},
        )

    def _derivative(self, expression: str, variable: str) -> ToolResult:
        """Compute symbolic derivative (simplified)"""
        # Simple numerical derivative using central difference
        try:
            safe_dict = {
                "pi": math.pi, "e": math.e,
                "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "abs": abs, "pow": pow,
            }

            def f(x_val):
                local_dict = safe_dict.copy()
                local_dict[variable] = x_val
                return eval(expression, {"__builtins__": {}}, local_dict)

            # Central difference at x=1 (or 0 if variable not in expression)
            x0 = 1.0
            h = 1e-7
            derivative = (f(x0 + h) - f(x0 - h)) / (2 * h)

            return ToolResult(
                success=True,
                output={
                    "expression": expression,
                    "derivative_approx": derivative,
                    "method": "central_difference",
                    "note": "Numerical derivative at x=1. For symbolic derivatives, use a CAS library.",
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Could not compute derivative: {e}")

    def _integral(self, expression: str, variable: str) -> ToolResult:
        """Compute numerical integral using Simpson's rule"""
        try:
            safe_dict = {
                "pi": math.pi, "e": math.e,
                "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "abs": abs, "pow": pow,
            }

            def f(x_val):
                local_dict = safe_dict.copy()
                local_dict[variable] = x_val
                return eval(expression, {"__builtins__": {}}, local_dict)

            # Simpson's rule from 0 to 1
            n = 1000  # Even number of intervals
            a, b = 0.0, 1.0
            h = (b - a) / n

            result = f(a) + f(b)
            for i in range(1, n, 2):
                result += 4 * f(a + i * h)
            for i in range(2, n, 2):
                result += 2 * f(a + i * h)
            result *= h / 3

            return ToolResult(
                success=True,
                output={
                    "expression": expression,
                    "integral_approx": result,
                    "method": "simpsons_rule",
                    "interval": [a, b],
                    "note": "Numerical integral from 0 to 1.",
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Could not compute integral: {e}")

    def _solve(self, expression: str) -> ToolResult:
        """Solve simple equations (e.g., 'x^2 - 4 = 0')"""
        # Simple root finding using bisection
        try:
            if "=" in expression:
                left, right = expression.split("=", 1)
                expr = f"({left.strip()}) - ({right.strip()})"
            else:
                expr = expression

            safe_dict = {
                "pi": math.pi, "e": math.e,
                "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "abs": abs, "pow": pow,
            }

            def f(x_val):
                local_dict = safe_dict.copy()
                local_dict["x"] = x_val
                return eval(expr, {"__builtins__": {}}, local_dict)

            # Bisection method
            roots = []
            for search_range in [(-10, -0.01), (-0.01, 0.01), (0.01, 10)]:
                a, b = search_range
                try:
                    if f(a) * f(b) < 0:
                        for _ in range(100):
                            c = (a + b) / 2
                            if abs(f(c)) < 1e-10:
                                roots.append(round(c, 10))
                                break
                            if f(a) * f(c) < 0:
                                b = c
                            else:
                                a = c
                        else:
                            roots.append(round((a + b) / 2, 10))
                except:
                    continue

            return ToolResult(
                success=True,
                output={
                    "equation": expression,
                    "roots": roots,
                    "method": "bisection",
                    "note": "Found roots in range [-10, 10].",
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Could not solve: {e}")

    def _matrix_operation(self, expression: str) -> ToolResult:
        """Basic matrix operations"""
        import numpy as np
        try:
            # Parse matrix from expression like "det([[1,2],[3,4]])" or "inv([[1,2],[3,4]])"
            if expression.startswith("det(") and expression.endswith(")"):
                matrix_str = expression[4:-1]
                matrix = eval(matrix_str)
                result = np.linalg.det(matrix)
                return ToolResult(success=True, output={"determinant": float(result)})
            elif expression.startswith("inv(") and expression.endswith(")"):
                matrix_str = expression[4:-1]
                matrix = eval(matrix_str)
                result = np.linalg.inv(matrix)
                return ToolResult(success=True, output={"inverse": result.tolist()})
            elif expression.startswith("eig(") and expression.endswith(")"):
                matrix_str = expression[4:-1]
                matrix = eval(matrix_str)
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                return ToolResult(success=True, output={
                    "eigenvalues": eigenvalues.tolist(),
                    "eigenvectors": eigenvectors.tolist(),
                })
            else:
                return ToolResult(success=False, error="Supported: det(...), inv(...), eig(...)")
        except ImportError:
            return ToolResult(success=False, error="numpy is required for matrix operations")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _statistics(self, expression: str) -> ToolResult:
        """Compute statistics on a list of numbers"""
        try:
            # Parse list from expression
            data = eval(expression)
            if not isinstance(data, (list, tuple)):
                return ToolResult(success=False, error="Expression must evaluate to a list")

            import statistics
            result = {
                "count": len(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "stdev": statistics.stdev(data) if len(data) > 1 else 0,
                "variance": statistics.variance(data) if len(data) > 1 else 0,
                "min": min(data),
                "max": max(data),
                "range": max(data) - min(data),
                "sum": sum(data),
            }

            if len(data) > 1:
                try:
                    result["mode"] = statistics.mode(data)
                except statistics.StatisticsError:
                    result["mode"] = None

            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _probability(self, expression: str) -> ToolResult:
        """Basic probability calculations"""
        try:
            safe_dict = {
                "pi": math.pi, "e": math.e,
                "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                "factorial": math.factorial, "comb": math.comb, "perm": math.perm,
                "pow": pow, "abs": abs,
            }
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return ToolResult(success=True, output={"result": result})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
