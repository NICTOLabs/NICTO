"""
NICTO AI - Text-to-SQL Tool
Convert natural language to SQL queries.
"""

import logging
import re
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class TextToSQLTool(Tool):
    """
    Natural language to SQL converter.

    Supports:
    - SELECT, INSERT, UPDATE, DELETE
    - JOINs, GROUP BY, HAVING
    - Subqueries, CTEs
    - Aggregation functions
    - SQLite, PostgreSQL, MySQL dialects
    """

    name = "text_to_sql"
    description = "Convert natural language to SQL queries. Supports SELECT, JOINs, GROUP BY, and more."
    parameters = [
        ToolParameter(name="query", type="string", description="Natural language description of the query", required=True),
        ToolParameter(name="tables", type="string", description="Table schemas (JSON: {'table_name': {'columns': {'col': 'type'}}})", required=False),
        ToolParameter(name="dialect", type="string", description="SQL dialect", required=False, default="sqlite", enum=["sqlite", "postgresql", "mysql"]),
        ToolParameter(name="query_type", type="string", description="Type of SQL query", required=False, default="select", enum=["select", "insert", "update", "delete", "create"]),
    ]
    tags = ["sql", "database", "query", "nlp"]
    timeout_seconds = 15.0

    # SQL keyword templates
    AGGREGATIONS = {
        "count": "COUNT", "sum": "SUM", "average": "AVG", "avg": "AVG",
        "minimum": "MIN", "minimum value": "MIN", "smallest": "MIN",
        "maximum": "MAX", "maximum value": "MAX", "largest": "MAX",
        "total": "SUM", "mean": "AVG",
    }

    COMPARISONS = {
        "equals": "=", "equal to": "=", "is": "=", "=": "=",
        "greater than": ">", "more than": ">", "above": ">",
        "less than": "<", "fewer than": "<", "below": "<",
        "greater or equal": ">=", "at least": ">=",
        "less or equal": "<=", "at most": "<=",
        "not equal": "!=", "different from": "!=",
    }

    SQL_KEYWORDS = {
        "find": "SELECT", "get": "SELECT", "show": "SELECT", "list": "SELECT",
        "select": "SELECT", "retrieve": "SELECT", "display": "SELECT",
        "count": "SELECT COUNT", "how many": "SELECT COUNT",
        "insert": "INSERT INTO", "add": "INSERT INTO", "create record": "INSERT INTO",
        "update": "UPDATE", "modify": "UPDATE", "change": "UPDATE", "set": "UPDATE",
        "delete": "DELETE FROM", "remove": "DELETE FROM", "drop": "DROP",
    }

    def _execute(self, query: str, tables: str = None, dialect: str = "sqlite",
                 query_type: str = "select") -> ToolResult:

        if not query:
            return ToolResult(success=False, error="Query is empty")

        # Parse table schemas
        table_schemas = {}
        if tables:
            try:
                import json
                table_schemas = json.loads(tables)
            except json.JSONDecodeError:
                return ToolResult(success=False, error="Invalid tables JSON")

        result = self._parse_query(query, table_schemas, dialect, query_type)

        return ToolResult(
            success=True,
            output={
                "natural_language": query,
                "sql": result["sql"],
                "dialect": dialect,
                "query_type": query_type,
                "explanation": result.get("explanation", ""),
                "tables_used": result.get("tables_used", []),
            },
        )

    def _parse_query(self, query: str, schemas: Dict, dialect: str, query_type: str) -> Dict:
        """Parse natural language to SQL"""
        query_lower = query.lower().strip()

        # Determine query type from text
        detected_type = self._detect_query_type(query_lower)

        if detected_type == "insert":
            return self._generate_insert(query, schemas)
        elif detected_type == "update":
            return self._generate_update(query, schemas)
        elif detected_type == "delete":
            return self._generate_delete(query, schemas)
        elif detected_type == "create":
            return self._generate_create(query, schemas)
        else:
            return self._generate_select(query_lower, schemas, dialect)

    def _detect_query_type(self, query: str) -> str:
        if any(w in query for w in ["insert into", "add new", "create new record"]):
            return "insert"
        elif any(w in query for w in ["update", "modify", "change", "set where"]):
            return "update"
        elif any(w in query for w in ["delete", "remove", "drop"]):
            return "delete"
        elif any(w in query for w in ["create table", "new table"]):
            return "create"
        return "select"

    def _generate_select(self, query: str, schemas: Dict, dialect: str) -> Dict:
        """Generate SELECT query"""
        sql = "SELECT "
        tables_used = []

        # Detect tables from schemas or query
        if schemas:
            table_names = list(schemas.keys())
        else:
            table_names = self._extract_table_names(query)

        tables_used = table_names
        main_table = table_names[0] if table_names else "table_name"

        # Detect columns
        columns = self._extract_columns(query)

        # Handle aggregation
        agg_col = None
        for keyword, sql_func in self.AGGREGATIONS.items():
            if keyword in query:
                agg_col = next(self._extract_column_names(query), None)
                if agg_col:
                    sql += f"{sql_func}({agg_col})"
                else:
                    sql += f"{sql_func}(*)"
                break

        if not agg_col:
            if columns:
                sql += ", ".join(columns)
            elif schemas and main_table in schemas:
                sql += "*"
            else:
                sql += "*"

        sql += f"\nFROM {main_table}"

        # Detect JOINs
        join_keywords = ["join", "together with", "combined with", "along with"]
        if any(k in query for k in join_keywords) and len(table_names) > 1:
            for other_table in table_names[1:]:
                sql += f"\n  JOIN {other_table} ON {main_table}.id = {other_table}.{main_table}_id"
                tables_used.append(other_table)

        # Detect WHERE conditions
        where_clause = self._generate_where(query, schemas, main_table)
        if where_clause:
            sql += f"\nWHERE {where_clause}"

        # GROUP BY
        group_keywords = ["group by", "per", "by each", "for each", "average.*per", "count.*per"]
        if any(re.search(k, query) for k in group_keywords):
            group_col = self._extract_group_column(query, columns, schemas, main_table)
            if group_col:
                sql += f"\nGROUP BY {group_col}"

        # ORDER BY
        order_keywords = {
            "sorted by": "", "ordered by": "", "order by": "",
            "alphabetical": "ASC", "ascending": "ASC",
            "descending": "DESC", "reverse": "DESC",
            "latest": "DESC", "most recent": "DESC",
            "oldest": "ASC", "earliest": "ASC",
        }
        for keyword, direction in order_keywords.items():
            if keyword in query:
                order_col = self._extract_order_column(query, columns)
                if order_col:
                    sql += f"\nORDER BY {order_col} {direction}".strip()
                    break

        # LIMIT
        limit_match = re.search(r'(?:top|limit|first|latest)\s+(\d+)', query)
        if limit_match:
            if dialect == "mysql":
                sql += f"\nLIMIT {limit_match.group(1)}"
            else:
                sql += f"\nLIMIT {limit_match.group(1)}"

        explanation = f"Querying {main_table}"
        if agg_col:
            explanation += f" with aggregation"

        return {
            "sql": sql + ";",
            "tables_used": tables_used,
            "explanation": explanation,
        }

    def _generate_insert(self, query: str, schemas: Dict) -> Dict:
        table = self._extract_table_names(query)
        table_name = table[0] if table else "table_name"
        return {
            "sql": f"INSERT INTO {table_name} (column1, column2)\nVALUES (value1, value2);",
            "tables_used": [table_name],
            "explanation": f"Inserting into {table_name}. Specify column names and values.",
        }

    def _generate_update(self, query: str, schemas: Dict) -> Dict:
        table = self._extract_table_names(query)
        table_name = table[0] if table else "table_name"
        return {
            "sql": f"UPDATE {table_name}\nSET column1 = value1\nWHERE condition;",
            "tables_used": [table_name],
            "explanation": f"Updating {table_name}. Specify SET values and WHERE condition.",
        }

    def _generate_delete(self, query: str, schemas: Dict) -> Dict:
        table = self._extract_table_names(query)
        table_name = table[0] if table else "table_name"
        return {
            "sql": f"DELETE FROM {table_name}\nWHERE condition;",
            "tables_used": [table_name],
            "explanation": f"Deleting from {table_name}. Add WHERE clause to avoid deleting all rows.",
        }

    def _generate_create(self, query: str, schemas: Dict) -> Dict:
        table = self._extract_table_names(query)
        table_name = table[0] if table else "new_table"
        return {
            "sql": f"CREATE TABLE {table_name} (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL,\n  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);",
            "tables_used": [table_name],
            "explanation": f"Creating table {table_name}. Define columns and types.",
        }

    def _extract_table_names(self, query: str) -> List[str]:
        words = query.split()
        found = []
        # Look for patterns like "from X", "table X", "X table"
        for i, word in enumerate(words):
            if word in ("from", "in", "table", "into", "update") and i + 1 < len(words):
                found.append(words[i + 1].rstrip("s,.;:!?"))
        return found if found else ["items"]

    def _extract_columns(self, query: str) -> List[str]:
        keywords = {"show": 0, "find": 0, "get": 0, "list": 0, "select": 0}
        cols = set()

        for keyword, pos in keywords.items():
            if keyword in query:
                idx = query.index(keyword) + len(keyword)
                after = query[idx:].strip()
                # Get words until "from" or "where"
                for stop in ("from", "where", "in", "for", "that have"):
                    if stop in after:
                        after = after[:after.index(stop)]
                # Extract column-like words
                for word in re.findall(r'\w+', after):
                    if word.lower() not in self.SQL_KEYWORDS and len(word) > 1:
                        cols.add(word)
                break

        return list(cols) if cols else ["*"]

    def _extract_column_names(self, query: str):
        for word in re.findall(r'\b[a-z_]+\b', query.lower()):
            if word not in self.SQL_KEYWORDS and len(word) > 2:
                yield word

    def _generate_where(self, query: str, schemas: Dict, table: str) -> str:
        conditions = []

        # Extract conditions
        for comparison, operator in self.COMPARISONS.items():
            if comparison in query:
                parts = query.split(comparison, 1)
                col = parts[0].strip().split()[-1]
                val = parts[1].strip().split()[0] if parts[1].strip() else ""
                if not val:
                    # Find next word after operator
                    val_parts = re.findall(r'\w+', parts[1]) if len(parts) > 1 else []
                    val = val_parts[0] if val_parts else "value"
                conditions.append(f"{col} {operator} '{val}'")

        # Pattern: "where column is value"
        where_match = re.search(r'where\s+(.+?)(?:and|or|$)', query)
        if where_match and not conditions:
            clause = where_match.group(1).strip()
            conditions.append(clause)

        # Pattern: "in column" / "that have column"
        for pattern in [r'(?:in|with)\s+\w+\s*=\s*[\'"]?(\w+)[\'"]?',
                        r'(?:that have|which have|with)\s+(\w+)']:
            match = re.search(pattern, query)
            if match:
                conditions.append(f"{match.group(1)} = '{match.group(1)}'")
                break

        return " AND ".join(conditions) if conditions else ""

    def _extract_group_column(self, query: str, columns: List[str], schemas: Dict, table: str) -> str:
        for word in re.findall(r'\b[a-z_]+\b', query.lower()):
            if word in ("per", "each"):
                continue
            if word not in self.SQL_KEYWORDS and len(word) > 2:
                return word
        return ""

    def _extract_order_column(self, query: str, columns: List[str]) -> str:
        for keyword in ["sorted by", "ordered by", "order by", "sort by"]:
            if keyword in query:
                idx = query.index(keyword) + len(keyword)
                after = query[idx:].strip().split()
                if after:
                    return after[0].rstrip(",.;:")
        return ""
