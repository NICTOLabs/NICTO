"""
NICTO AI - Knowledge Tool
Query, crawl, and ingest info into NICTO's knowledge base.
"""

import json
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class KnowledgeTool(Tool):
    """
    Query NICTO's internal knowledge base.

    Searches indexed documents, web crawl results, and stored facts.
    Supports sub-commands: query, crawl, ingest, stats
    """

    name = "knowledge_query"
    description = "Query NICTO's knowledge base for stored information. Supports sub-commands: query, crawl, ingest, stats."
    parameters = [
        ToolParameter(name="query", type="string", description="Search query or 'crawl:URL' to crawl, 'ingest:text' to add text, 'stats' for KB stats", required=True),
        ToolParameter(name="top_k", type="integer", description="Number of results to return", required=False, default=5),
        ToolParameter(name="title", type="string", description="Title (for ingest)", required=False),
        ToolParameter(name="url", type="string", description="Source URL (for ingest)", required=False),
    ]
    tags = ["knowledge", "search", "retrieval", "crawl"]
    timeout_seconds = 30.0

    def __init__(self, knowledge_base=None):
        self._kb = knowledge_base
        self._kb_started = False

    def _ensure_kb(self):
        """Lazily init and start the KnowledgeBase if needed."""
        if self._kb is not None and not self._kb_started:
            self._kb.start_sync()
            self._kb_started = True
        if self._kb is None:
            from nicto_ai.knowledge import KnowledgeBase
            self._kb = KnowledgeBase()
            self._kb.start_sync()
            self._kb_started = True

    def _execute(self, query: str, top_k: int = 5, title: str = None, url: str = None) -> ToolResult:
        try:
            self._ensure_kb()

            # Check for sub-commands
            if query.lower() == "stats":
                stats = self._kb.stats
                return ToolResult(
                    success=True,
                    output={
                        "total_entries": stats.total_entries,
                        "index_path": stats.index_path,
                    },
                    metadata={"action": "stats"},
                )

            if query.lower().startswith("crawl:"):
                crawl_url = query[6:].strip()
                result = self._kb.crawl_sync(crawl_url)
                return ToolResult(
                    success=True,
                    output=result,
                    metadata={"action": "crawl", "url": crawl_url},
                )

            if query.lower().startswith("ingest:"):
                text = query[7:].strip()
                self._kb.ingest_text(text=text, title=title or "", url=url or "")
                return ToolResult(
                    success=True,
                    output={"ingested": True, "text_length": len(text)},
                    metadata={"action": "ingest"},
                )

            # Default: search knowledge base
            query_result = self._kb.query(query, top_k=top_k)

            formatted_results = []
            for r in query_result.results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "summary": r.get("summary", ""),
                    "text": r.get("summary", ""),
                    "score": r.get("score", 0),
                    "source": "knowledge_base",
                })

            return ToolResult(
                success=True,
                output=formatted_results,
                metadata={
                    "query": query,
                    "result_count": len(formatted_results),
                    "query_time_ms": query_result.query_time_ms,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
