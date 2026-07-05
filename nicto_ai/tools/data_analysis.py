"""
NICTO AI - Data Analysis Tool
Analyze CSV/JSON data, generate statistics, and create visualizations.
"""

import json
import csv
import os
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class DataAnalysisTool(Tool):
    """
    Data analysis for CSV and JSON files.

    Supports:
    - Summary statistics
    - Column analysis
    - Correlation detection
    - Data quality checks
    - Pattern recognition
    """

    name = "data_analysis"
    description = "Analyze data from CSV or JSON files. Provides statistics, correlations, and insights."
    parameters = [
        ToolParameter(name="file_path", type="string", description="Path to CSV or JSON file", required=True),
        ToolParameter(name="operation", type="string", description="Analysis type", required=False, default="summary", enum=["summary", "columns", "correlations", "quality", "filter", "group"]),
        ToolParameter(name="column", type="string", description="Column name for column-specific operations", required=False),
        ToolParameter(name="filter_expr", type="string", description="Filter expression (column>value)", required=False),
    ]
    tags = ["data", "analysis", "csv", "json", "statistics"]
    timeout_seconds = 30.0

    def _execute(self, file_path: str, operation: str = "summary", column: str = None, filter_expr: str = None) -> ToolResult:
        if not os.path.exists(file_path):
            return ToolResult(success=False, error=f"File not found: {file_path}")

        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".csv":
                data = self._load_csv(file_path)
            elif ext == ".json":
                data = self._load_json(file_path)
            else:
                return ToolResult(success=False, error=f"Unsupported file type: {ext}")

            if not data:
                return ToolResult(success=False, error="No data found in file")

            if operation == "summary":
                return self._summary(data)
            elif operation == "columns":
                return self._column_analysis(data, column)
            elif operation == "correlations":
                return self._correlations(data)
            elif operation == "quality":
                return self._data_quality(data)
            elif operation == "filter":
                return self._filter_data(data, filter_expr)
            elif operation == "group":
                return self._group_data(data, column)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _load_csv(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _load_json(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            return []

    def _summary(self, data: List[Dict]) -> ToolResult:
        if not data:
            return ToolResult(success=True, output={"rows": 0, "columns": 0})

        columns = list(data[0].keys())
        numeric_cols = []
        categorical_cols = []

        for col in columns:
            values = [row.get(col) for row in data if row.get(col) is not None]
            try:
                float_vals = [float(v) for v in values[:100]]
                numeric_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)

        summary = {
            "total_rows": len(data),
            "total_columns": len(columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "columns": {},
        }

        for col in numeric_cols:
            values = []
            for row in data:
                try:
                    values.append(float(row[col]))
                except (ValueError, TypeError, KeyError):
                    continue
            if values:
                summary["columns"][col] = {
                    "type": "numeric",
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "unique": len(set(values)),
                }

        for col in categorical_cols:
            values = [row[col] for row in data if col in row and row[col] is not None]
            if values:
                from collections import Counter
                counts = Counter(values)
                summary["columns"][col] = {
                    "type": "categorical",
                    "count": len(values),
                    "unique": len(set(values)),
                    "top_values": dict(counts.most_common(5)),
                }

        return ToolResult(success=True, output=summary)

    def _column_analysis(self, data: List[Dict], column: str) -> ToolResult:
        if not column:
            return ToolResult(success=False, error="Column name is required")

        values = [row.get(column) for row in data if row.get(column) is not None]
        if not values:
            return ToolResult(success=False, error=f"Column '{column}' not found or empty")

        try:
            numeric_vals = [float(v) for v in values]
            return ToolResult(
                success=True,
                output={
                    "column": column,
                    "type": "numeric",
                    "count": len(numeric_vals),
                    "mean": sum(numeric_vals) / len(numeric_vals),
                    "min": min(numeric_vals),
                    "max": max(numeric_vals),
                    "std": (sum((x - sum(numeric_vals)/len(numeric_vals))**2 for x in numeric_vals) / len(numeric_vals)) ** 0.5,
                    "unique": len(set(numeric_vals)),
                },
            )
        except (ValueError, TypeError):
            from collections import Counter
            counts = Counter(values)
            return ToolResult(
                success=True,
                output={
                    "column": column,
                    "type": "categorical",
                    "count": len(values),
                    "unique": len(set(values)),
                    "top_values": dict(counts.most_common(10)),
                    "null_count": len(data) - len(values),
                },
            )

    def _correlations(self, data: List[Dict]) -> ToolResult:
        columns = list(data[0].keys()) if data else []
        numeric_cols = []
        for col in columns:
            try:
                [float(row[col]) for row in data[:10] if row.get(col)]
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue

        correlations = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                vals1, vals2 = [], []
                for row in data:
                    try:
                        v1 = float(row[col1])
                        v2 = float(row[col2])
                        vals1.append(v1)
                        vals2.append(v2)
                    except (ValueError, TypeError, KeyError):
                        continue

                if len(vals1) > 1:
                    n = len(vals1)
                    mean1 = sum(vals1) / n
                    mean2 = sum(vals2) / n
                    cov = sum((a - mean1) * (b - mean2) for a, b in zip(vals1, vals2)) / n
                    std1 = (sum((x - mean1) ** 2 for x in vals1) / n) ** 0.5
                    std2 = (sum((x - mean2) ** 2 for x in vals2) / n) ** 0.5
                    if std1 > 0 and std2 > 0:
                        corr = cov / (std1 * std2)
                        correlations[f"{col1} vs {col2}"] = round(corr, 4)

        return ToolResult(success=True, output={"correlations": correlations, "numeric_columns": numeric_cols})

    def _data_quality(self, data: List[Dict]) -> ToolResult:
        if not data:
            return ToolResult(success=True, output={"quality": "empty"})

        columns = list(data[0].keys())
        quality = {}
        for col in columns:
            total = len(data)
            null_count = sum(1 for row in data if not row.get(col))
            unique_vals = len(set(row.get(col) for row in data if row.get(col)))
            quality[col] = {
                "null_count": null_count,
                "null_pct": round(null_count / total * 100, 1) if total > 0 else 0,
                "unique_values": unique_vals,
                "completeness": round((total - null_count) / total * 100, 1) if total > 0 else 0,
            }

        return ToolResult(success=True, output={"total_rows": len(data), "columns": quality})

    def _filter_data(self, data: List[Dict], filter_expr: str) -> ToolResult:
        if not filter_expr:
            return ToolResult(success=False, error="Filter expression is required")

        # Simple filter: "column>value" or "column=value" or "column<value"
        import operator
        ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "=": operator.eq, "==": operator.eq}

        for op_str, op_func in ops.items():
            if op_str in filter_expr:
                col, val = filter_expr.split(op_str, 1)
                col, val = col.strip(), val.strip()
                try:
                    val_num = float(val)
                    filtered = [row for row in data if col in row and op_func(float(row[col]), val_num)]
                except ValueError:
                    filtered = [row for row in data if col in row and op_func(row[col], val)]
                return ToolResult(success=True, output=filtered[:100], metadata={"filtered_count": len(filtered)})

        return ToolResult(success=False, error="Invalid filter expression. Use: column>value, column<value, column=value")

    def _group_data(self, data: List[Dict], column: str) -> ToolResult:
        if not column:
            return ToolResult(success=False, error="Column name is required")

        groups = {}
        for row in data:
            key = row.get(column, "null")
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        result = {k: len(v) for k, v in groups.items()}
        return ToolResult(success=True, output={"column": column, "groups": result, "group_count": len(result)})
