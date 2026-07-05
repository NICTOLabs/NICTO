"""
NICTO AI - Knowledge Tool
Query NICTO's knowledge base for stored information.
"""

import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class KnowledgeTool(Tool):
    """
    Query NICTO's internal knowledge base.

    Searches indexed documents, web crawl results, and stored facts.
    """

    name = "knowledge_query"
    description = "Query NICTO's knowledge base for stored information. Search indexed documents and facts."
    parameters = [
        ToolParameter(name="query", type="string", description="Search query", required=True),
        ToolParameter(name="top_k", type="integer", description="Number of results to return", required=False, default=5),
        ToolParameter(name="source_filter", type="string", description="Filter by source type (web, document, manual)", required=False),
    ]
    tags = ["knowledge", "search", "retrieval"]
    timeout_seconds = 10.0

    def __init__(self, knowledge_base=None):
        self._kb = knowledge_base

    def _execute(self, query: str, top_k: int = 5, source_filter: str = None) -> ToolResult:
        if self._kb is None:
            return ToolResult(
                success=False,
                error="Knowledge base not initialized. Use NICTOKnowledgeBase first.",
            )

        try:
            results = self._kb.query(query, top_k=top_k)

            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "text": r.get("text", "")[:500],
                    "score": r.get("score", 0),
                    "source": r.get("source", "unknown"),
                })

            return ToolResult(
                success=True,
                output=formatted_results,
                metadata={
                    "query": query,
                    "result_count": len(formatted_results),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
