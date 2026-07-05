"""
NICTO AI - Web Search Tool
Search the internet for information, news, and documentation.
"""

import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter, ToolPermission

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """
    Search the web for information.

    Uses NICTO's browser system to perform web searches
    and return structured results.
    """

    name = "web_search"
    description = "Search the internet for information. Returns relevant web pages with titles, URLs, and snippets."
    parameters = [
        ToolParameter(name="query", type="string", description="Search query", required=True),
        ToolParameter(name="max_results", type="integer", description="Maximum results to return (1-10)", required=False, default=5),
        ToolParameter(name="engine", type="string", description="Search engine to use", required=False, default="bing", enum=["bing", "google", "duckduckgo"]),
    ]
    permissions = [ToolPermission.NETWORK]
    tags = ["search", "web", "information"]
    timeout_seconds = 15.0

    def __init__(self, browser=None):
        self._browser = browser

    def _execute(self, query: str, max_results: int = 5, engine: str = "bing") -> ToolResult:
        if self._browser is None:
            return ToolResult(
                success=False,
                error="Web search requires a browser instance. Initialize NICTOBrowser first.",
            )

        try:
            response = self._browser.search(query, max_results=max_results, engine=engine)

            results = []
            for r in response.results[:max_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", ""),
                    "rank": r.get("rank", 0),
                })

            return ToolResult(
                success=True,
                output=results,
                metadata={
                    "query": query,
                    "engine": engine,
                    "result_count": len(results),
                    "search_time_ms": response.search_time_ms,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
