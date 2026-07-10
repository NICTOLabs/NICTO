"""
NICTO AI - Markdown Builder Tool
Generate Markdown documents, tables, and formatting.
"""

import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class MarkdownBuilderTool(Tool):
    """
    Markdown document generator.

    Features:
    - Generate complete documents
    - Tables
    - Code blocks
    - Lists (ordered, unordered)
    - Headers and formatting
    - Links and images
    - Task lists
    """

    name = "markdown_builder"
    description = "Build Markdown documents: headers, tables, code blocks, lists, links, and complete docs."
    parameters = [
        ToolParameter(name="operation", type="string", description="Operation to perform", required=True, enum=[
            "document", "table", "code_block", "list", "header", "link",
            "task_list", "blockquote", "horizontal_rule", "readme",
        ]),
        ToolParameter(name="content", type="string", description="Main content or title", required=True),
        ToolParameter(name="items", type="string", description="Items for lists/table (JSON array or newline-separated)", required=False),
        ToolParameter(name="level", type="integer", description="Header level (1-6)", required=False, default=1),
        ToolParameter(name="language", type="string", description="Programming language for code blocks", required=False),
        ToolParameter(name="columns", type="string", description="Table columns (comma-separated)", required=False),
    ]
    tags = ["markdown", "document", "formatting", "readme"]
    timeout_seconds = 10.0

    def _execute(self, operation: str, content: str, items: str = None,
                 level: int = 1, language: str = None, columns: str = None, **kwargs) -> ToolResult:

        level = max(1, min(6, level))

        if operation == "document":
            return self._build_document(content, items)
        elif operation == "table":
            return self._build_table(content, items, columns)
        elif operation == "code_block":
            return self._build_code_block(content, language)
        elif operation == "list":
            return self._build_list(content, items, ordered=kwargs.get("ordered", False))
        elif operation == "header":
            return self._build_header(content, level)
        elif operation == "link":
            return self._build_link(content, kwargs.get("url", ""), kwargs.get("title", ""))
        elif operation == "task_list":
            return self._build_task_list(items)
        elif operation == "blockquote":
            return self._build_blockquote(content)
        elif operation == "horizontal_rule":
            return self._build_hr()
        elif operation == "readme":
            return self._build_readme(content, items)
        else:
            return ToolResult(success=False, error=f"Unknown operation: {operation}")

    def _build_document(self, title: str, sections: str = None) -> ToolResult:
        """Build a complete Markdown document"""
        doc = f"# {title}\n\n"

        if sections:
            section_list = [s.strip() for s in sections.split('\n') if s.strip()]
            for section in section_list:
                doc += f"## {section}\n\nDescription of {section.lower()}.\n\n"
        else:
            doc += "## Overview\n\nOverview content here.\n\n"
            doc += "## Features\n\n- Feature 1\n- Feature 2\n- Feature 3\n\n"
            doc += "## Getting Started\n\nInstructions here.\n\n"
            doc += "## License\n\nMIT License\n"

        return ToolResult(success=True, output={"markdown": doc})

    def _build_table(self, caption: str, rows: str = None, columns: str = None) -> ToolResult:
        """Build a Markdown table"""
        if columns:
            cols = [c.strip() for c in columns.split(',')]
        else:
            cols = ["Column 1", "Column 2", "Column 3"]

        # Header
        table = "| " + " | ".join(cols) + " |\n"
        table += "|" + "|".join(["---"] * len(cols)) + "|\n"

        # Rows
        if rows:
            row_list = [r.strip() for r in rows.split('\n') if r.strip()]
            for row in row_list:
                values = [v.strip() for v in row.split(',')]
                # Pad or truncate to match columns
                while len(values) < len(cols):
                    values.append("")
                table += "| " + " | ".join(values[:len(cols)]) + " |\n"
        else:
            # Sample row
            table += "| " + " | ".join(["..."] * len(cols)) + " |\n"

        if caption:
            table = f"**{caption}**\n\n{table}"

        return ToolResult(success=True, output={"markdown": table})

    def _build_code_block(self, code: str, language: str = None) -> ToolResult:
        """Build a code block"""
        lang = language or ""
        block = f"```{lang}\n{code}\n```"
        return ToolResult(success=True, output={"markdown": block})

    def _build_list(self, title: str, items: str = None, ordered: bool = False) -> ToolResult:
        """Build a list"""
        result = ""
        if title:
            result = f"### {title}\n\n"

        if items:
            item_list = [i.strip() for i in items.split('\n') if i.strip()]
            for idx, item in enumerate(item_list, 1):
                if ordered:
                    result += f"{idx}. {item}\n"
                else:
                    result += f"- {item}\n"
        else:
            result += "- Item 1\n- Item 2\n- Item 3\n"

        return ToolResult(success=True, output={"markdown": result})

    def _build_header(self, text: str, level: int = 1) -> ToolResult:
        """Build a header"""
        prefix = "#" * level
        header = f"{prefix} {text}"
        return ToolResult(success=True, output={"markdown": header})

    def _build_link(self, text: str, url: str, title: str = "") -> ToolResult:
        """Build a link"""
        if title:
            link = f"[{text}]({url} \"{title}\")"
        else:
            link = f"[{text}]({url})"
        return ToolResult(success=True, output={"markdown": link})

    def _build_task_list(self, items: str = None) -> ToolResult:
        """Build a task list"""
        result = ""
        if items:
            item_list = [i.strip() for i in items.split('\n') if i.strip()]
            for item in item_list:
                if item.startswith('[x]') or item.startswith('[X]'):
                    result += f"- [x] {item[3:].strip()}\n"
                elif item.startswith('[ ]'):
                    result += f"- [ ] {item[3:].strip()}\n"
                else:
                    result += f"- [ ] {item}\n"
        else:
            result = "- [x] Completed task\n- [ ] Pending task\n- [ ] Another task\n"

        return ToolResult(success=True, output={"markdown": result})

    def _build_blockquote(self, text: str) -> ToolResult:
        """Build a blockquote"""
        lines = text.split('\n')
        quote = '\n'.join(f"> {line}" for line in lines)
        return ToolResult(success=True, output={"markdown": quote})

    def _build_hr(self) -> ToolResult:
        """Build a horizontal rule"""
        return ToolResult(success=True, output={"markdown": "\n---\n"})

    def _build_readme(self, project_name: str, features: str = None) -> ToolResult:
        """Build a README template"""
        readme = f"""# {project_name}

> A brief description of {project_name}.

## Features

"""
        if features:
            feature_list = [f.strip() for f in features.split('\n') if f.strip()]
            for feat in feature_list:
                readme += f"- {feat}\n"
        else:
            readme += """- Feature 1
- Feature 2
- Feature 3
"""

        readme += """
## Installation

```bash
pip install """ + project_name.lower().replace(' ', '-') + """
```

## Quick Start

```python
import """ + project_name.lower().replace(' ', '_') + """

# Example usage
```

## Documentation

See the [docs](./docs) folder for detailed documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
"""
        return ToolResult(success=True, output={"markdown": readme})
