"""
NICTO AI - JSON Builder Tool
Generate, validate, transform, and query JSON data.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class JsonBuilderTool(Tool):
    """
    JSON manipulation engine.

    Features:
    - Generate JSON from schema
    - Validate JSON strings
    - Transform JSON structures
    - Query with JSONPath-like syntax
    - Merge JSON objects
    - Convert between formats
    """

    name = "json_builder"
    description = "Generate, validate, transform, and query JSON data. Supports schemas, merging, and format conversion."
    parameters = [
        ToolParameter(name="operation", type="string", description="Operation to perform", required=True, enum=[
            "validate", "generate", "transform", "query", "merge", "format", "minify",
        ]),
        ToolParameter(name="data", type="string", description="JSON data or schema (string)", required=False),
        ToolParameter(name="schema", type="string", description="JSON schema for generation", required=False),
        ToolParameter(name="query_path", type="string", description="JSONPath query", required=False),
        ToolParameter(name="transform_type", type="string", description="Transform type", required=False, enum=[
            "flatten", "unflatten", "rename_keys", "filter_keys", "wrap", "unwrap",
        ]),
    ]
    tags = ["json", "data", "transform", "validation", "query"]
    timeout_seconds = 10.0

    def _execute(self, operation: str, data: str = None, schema: str = None,
                 query_path: str = None, transform_type: str = None, **kwargs) -> ToolResult:

        if operation == "validate":
            return self._validate(data)
        elif operation == "generate":
            return self._generate(schema)
        elif operation == "transform":
            return self._transform(data, transform_type, **kwargs)
        elif operation == "query":
            return self._query(data, query_path)
        elif operation == "merge":
            return self._merge(data, **kwargs)
        elif operation == "format":
            return self._format(data)
        elif operation == "minify":
            return self._minify(data)
        else:
            return ToolResult(success=False, error=f"Unknown operation: {operation}")

    def _validate(self, data: str) -> ToolResult:
        """Validate JSON string"""
        if not data:
            return ToolResult(success=False, error="No data provided")

        try:
            parsed = json.loads(data)
            return ToolResult(
                success=True,
                output={
                    "valid": True,
                    "type": type(parsed).__name__,
                    "size_bytes": len(data.encode('utf-8')),
                    "preview": str(parsed)[:200],
                },
            )
        except json.JSONDecodeError as e:
            return ToolResult(
                success=True,
                output={
                    "valid": False,
                    "error": str(e),
                    "line": e.lineno,
                    "column": e.colno,
                },
            )

    def _generate(self, schema: str = None) -> ToolResult:
        """Generate JSON from schema"""
        if not schema:
            # Generate sample JSON
            sample = {
                "name": "example",
                "version": "1.0.0",
                "items": [
                    {"id": 1, "name": "Item 1", "active": True},
                    {"id": 2, "name": "Item 2", "active": False},
                ],
                "metadata": {
                    "created": "2024-01-01",
                    "author": "NICTO",
                },
            }
            return ToolResult(
                success=True,
                output={
                    "json": json.dumps(sample, indent=2),
                    "note": "Generated sample JSON. Provide schema for custom generation.",
                },
            )

        try:
            schema_obj = json.loads(schema)
            generated = self._generate_from_schema(schema_obj)
            return ToolResult(
                success=True,
                output={
                    "json": json.dumps(generated, indent=2),
                    "schema_used": schema_obj,
                },
            )
        except json.JSONDecodeError:
            return ToolResult(success=False, error="Invalid schema JSON")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _generate_from_schema(self, schema: Dict) -> Any:
        """Recursively generate JSON from schema"""
        if "type" not in schema:
            return {}

        type_name = schema["type"]

        if type_name == "string":
            if "enum" in schema:
                return schema["enum"][0]
            return schema.get("default", "string")
        elif type_name == "integer":
            return schema.get("default", 0)
        elif type_name == "number":
            return schema.get("default", 0.0)
        elif type_name == "boolean":
            return schema.get("default", True)
        elif type_name == "array":
            if "items" in schema:
                return [self._generate_from_schema(schema["items"])]
            return []
        elif type_name == "object":
            obj = {}
            for prop, prop_schema in schema.get("properties", {}).items():
                obj[prop] = self._generate_from_schema(prop_schema)
            return obj
        return {}

    def _transform(self, data: str, transform_type: str = None, **kwargs) -> ToolResult:
        """Transform JSON structure"""
        if not data:
            return ToolResult(success=False, error="No data provided")

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON: {e}")

        if transform_type == "flatten":
            result = self._flatten_json(parsed)
        elif transform_type == "unflatten":
            result = self._unflatten_json(parsed)
        elif transform_type == "rename_keys":
            mapping = kwargs.get("mapping", {})
            result = self._rename_keys(parsed, mapping)
        elif transform_type == "filter_keys":
            keys = kwargs.get("keys", [])
            result = self._filter_keys(parsed, keys)
        elif transform_type == "wrap":
            wrapper = kwargs.get("wrapper_key", "data")
            result = {wrapper: parsed}
        elif transform_type == "unwrap":
            if isinstance(parsed, dict) and len(parsed) == 1:
                result = list(parsed.values())[0]
            else:
                result = parsed
        else:
            return ToolResult(success=False, error=f"Unknown transform: {transform_type}")

        return ToolResult(
            success=True,
            output={
                "transform": transform_type,
                "result": json.dumps(result, indent=2),
            },
        )

    def _flatten_json(self, data: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """Flatten nested JSON"""
        items = {}
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_json(v, new_key, sep))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(self._flatten_json(item, f"{new_key}[{i}]", sep))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = v
        return items

    def _unflatten_json(self, data: Dict, sep: str = ".") -> Dict:
        """Unflatten a flat JSON"""
        result = {}
        for key, value in data.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result

    def _rename_keys(self, data: Any, mapping: Dict) -> Any:
        """Rename keys in JSON"""
        if isinstance(data, dict):
            return {mapping.get(k, k): self._rename_keys(v, mapping) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._rename_keys(item, mapping) for item in data]
        return data

    def _filter_keys(self, data: Any, keys: List[str]) -> Any:
        """Filter to only specified keys"""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in keys}
        elif isinstance(data, list):
            return [self._filter_keys(item, keys) for item in data]
        return data

    def _query(self, data: str, query_path: str = None) -> ToolResult:
        """Query JSON with simple path"""
        if not data:
            return ToolResult(success=False, error="No data provided")

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON: {e}")

        if not query_path:
            return ToolResult(
                success=True,
                output={"data": parsed, "note": "No query path - returned full data"},
            )

        # Simple dot-notation query
        try:
            current = parsed
            for part in query_path.split('.'):
                if '[' in part and ']' in part:
                    # Array indexing
                    key, idx = part.split('[')
                    idx = int(idx.rstrip(']'))
                    current = current[key][idx]
                else:
                    current = current[part]

            return ToolResult(
                success=True,
                output={
                    "query": query_path,
                    "result": json.dumps(current, indent=2) if isinstance(current, (dict, list)) else current,
                },
            )
        except (KeyError, IndexError, TypeError) as e:
            return ToolResult(success=False, error=f"Query failed: {e}")

    def _merge(self, data: str, **kwargs) -> ToolResult:
        """Merge multiple JSON objects"""
        data2 = kwargs.get("data2")
        if not data or not data2:
            return ToolResult(success=False, error="Two JSON objects required")

        try:
            obj1 = json.loads(data)
            obj2 = json.loads(data2)
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON: {e}")

        def deep_merge(a, b):
            result = a.copy()
            for key, value in b.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(obj1, obj2)
        return ToolResult(
            success=True,
            output={"merged": json.dumps(merged, indent=2)},
        )

    def _format(self, data: str) -> ToolResult:
        """Pretty-print JSON"""
        try:
            parsed = json.loads(data)
            formatted = json.dumps(parsed, indent=2, sort_keys=True)
            return ToolResult(success=True, output={"formatted": formatted})
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON: {e}")

    def _minify(self, data: str) -> ToolResult:
        """Minify JSON"""
        try:
            parsed = json.loads(data)
            minified = json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
            return ToolResult(success=True, output={"minified": minified})
        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON: {e}")
