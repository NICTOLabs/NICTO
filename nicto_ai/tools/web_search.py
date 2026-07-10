"""
NICTO AI - Web Search Tool
Search the internet for information, news, and documentation via HTTP.
"""

import re
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter, ToolPermission

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """
    Search the web for information via Bing.

    Uses curl_cffi (TLS fingerprint impersonation) to reliably
    fetch search results without triggering bot detection.
    """

    name = "web_search"
    description = "Search the internet for information. Returns relevant web pages with titles, URLs, and snippets."
    parameters = [
        ToolParameter(name="query", type="string", description="Search query", required=True),
        ToolParameter(name="max_results", type="integer", description="Maximum results to return (1-10)", required=False, default=5),
        ToolParameter(name="engine", type="string", description="Search engine to use", required=False, default="bing", enum=["bing", "duckduckgo"]),
    ]
    permissions = [ToolPermission.NETWORK]
    tags = ["search", "web", "information"]
    timeout_seconds = 20.0

    def _execute(self, query: str, max_results: int = 5, engine: str = "bing") -> ToolResult:
        try:
            results = self._search_http(query, engine, max_results)
            formatted = []
            for i, r in enumerate(results[:max_results]):
                formatted.append({
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r.get("snippet", ""),
                    "rank": i + 1,
                })

            return ToolResult(
                success=True,
                output=formatted,
                metadata={
                    "query": query,
                    "engine": engine,
                    "result_count": len(formatted),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _search_http(self, query: str, engine: str, max_results: int) -> list:
        """Search via HTTP with TLS fingerprint impersonation."""
        from curl_cffi import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        if engine == "bing":
            url = f"https://www.bing.com/search?q={query.replace(' ', '+')}&count={max_results}"
        else:
            url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"

        resp = requests.get(url, headers=headers, impersonate="chrome131", timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        if engine == "bing":
            return self._parse_bing(soup)
        else:
            return self._parse_ddg(soup)

    def _parse_bing(self, soup) -> list:
        """Extract search results from Bing HTML."""
        results = []
        seen_urls = set()

        for a_tag in soup.find_all("a", class_="tilk"):
            text = a_tag.get_text(strip=True)
            url_match = re.search(r'https?://[^\s\u00A0<>]+', text)
            if not url_match:
                continue
            url = url_match.group()
            if url in seen_urls or "bing.com" in url or "microsoft.com" in url:
                continue
            seen_urls.add(url)

            h2 = a_tag.find_parent("h2")
            title = h2.get_text(strip=True) if h2 else text.split("https://")[0].strip()
            title = re.sub(r'\s+', ' ', title).strip()[:120]

            results.append({"title": title, "url": url})

        return results

    def _parse_ddg(self, soup) -> list:
        """Extract search results from DDG Lite HTML."""
        results = []
        seen_urls = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if not href.startswith("http") or "duckduckgo.com" in href:
                continue
            if href in seen_urls or not text:
                continue
            seen_urls.add(href)
            results.append({"title": text[:120], "url": href})

        return results
