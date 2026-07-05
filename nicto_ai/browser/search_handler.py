"""
NICTO AI - Search Handler
Search engine integration for autonomous web searching
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .engine import BrowserEngine, PageInfo
from .page_parser import PageParser, ParsedContent

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with extracted content"""
    title: str
    url: str
    snippet: str
    content: Optional[ParsedContent] = None
    rank: int = 0
    relevance_score: float = 0.0


@dataclass
class SearchResponse:
    """Full search response with results and metadata"""
    query: str
    engine: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_ms: float = 0.0


class SearchHandler:
    """
    Handles web searches through multiple search engines.

    Uses the BrowserEngine to perform actual searches, then
    parses results into structured SearchResponse objects.
    """

    def __init__(self, browser: BrowserEngine, parser: Optional[PageParser] = None):
        self.browser = browser
        self.parser = parser or PageParser()

    async def search(
        self,
        query: str,
        engine: str = "bing",
        max_results: int = 10,
        fetch_content: bool = False,
    ) -> SearchResponse:
        """
        Perform a web search and extract results.

        Args:
            query: Search query
            engine: Search engine ("duckduckgo", "google", "bing")
            max_results: Maximum results to return
            fetch_content: Whether to visit each result URL and extract full content

        Returns:
            SearchResponse with ranked results
        """
        import time
        start = time.time()

        # Build search URL
        urls = {
            "duckduckgo": f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}",
            "google": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "bing": f"https://www.bing.com/search?q={query.replace(' ', '+')}",
        }
        url = urls.get(engine, urls["duckduckgo"])

        # Navigate to search engine
        page_info = await self.browser.navigate(url)
        parsed = self.parser.parse_html(page_info.html, page_info.url)
        # Use raw text content for JS-rendered search results
        parsed.main_text = page_info.text_content

        # Extract search results from the parsed content
        results = self._extract_search_results(parsed, engine)

        # Optionally fetch full content for each result
        if fetch_content:
            for result in results[:max_results]:
                try:
                    result_page = await self.browser.navigate(result.url)
                    result.content = self.parser.parse_html(result_page.html, result.url)
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", result.url, e)

        elapsed = (time.time() - start) * 1000

        return SearchResponse(
            query=query,
            engine=engine,
            results=results[:max_results],
            total_results=len(results),
            search_time_ms=elapsed,
        )

    async def multi_search(
        self,
        query: str,
        engines: List[str] = None,
        max_results: int = 5,
    ) -> SearchResponse:
        """
        Search across multiple engines and merge results.

        Args:
            query: Search query
            engines: List of search engines to use
            max_results: Max results per engine

        Returns:
            Merged SearchResponse with deduplicated results
        """
        if engines is None:
            engines = ["bing", "google"]

        all_results = []
        for engine in engines:
            try:
                response = await self.search(query, engine, max_results)
                all_results.extend(response.results)
            except Exception as e:
                logger.warning("Search failed on %s: %s", engine, e)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return SearchResponse(
            query=query,
            engine=",".join(engines),
            results=unique_results,
            total_results=len(unique_results),
        )

    async def deep_search(
        self,
        query: str,
        depth: int = 2,
        max_results: int = 5,
    ) -> SearchResponse:
        """
        Deep search: search, then follow links from top results.

        Args:
            query: Initial search query
            depth: How many levels of link-following
            max_results: Results per level

        Returns:
            SearchResponse with deep results
        """
        # Initial search
        response = await self.search(query, max_results=max_results, fetch_content=True)

        if depth <= 1:
            return response

        # Follow links from top results
        extra_results = []
        for result in response.results[:3]:  # Follow top 3 results
            if result.content and result.content.links:
                for link in result.content.links[:2]:  # Follow top 2 links each
                    try:
                        page = await self.browser.navigate(link["href"])
                        content = self.parser.parse_html(page.html, page.url)
                        self.parser.rank_relevance(content, query)
                        if content.relevance_score > 0.1:
                            extra_results.append(SearchResult(
                                title=content.title,
                                url=page.url,
                                snippet=content.main_text[:200],
                                content=content,
                                relevance_score=content.relevance_score,
                            ))
                    except Exception as e:
                        logger.warning("Deep search link failed: %s", e)

        response.results.extend(extra_results)
        response.total_results = len(response.results)
        return response

    def _extract_search_results(
        self,
        parsed: ParsedContent,
        engine: str,
    ) -> List[SearchResult]:
        """
        Extract search results from parsed search engine page.

        Different engines have different result formats.
        """
        results = []

        if engine == "duckduckgo":
            results = self._extract_ddg_results(parsed)
        elif engine == "google":
            results = self._extract_google_results(parsed)
        elif engine == "bing":
            results = self._extract_bing_results(parsed)

        # Rank by relevance
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def _extract_ddg_results(self, parsed: ParsedContent) -> List[SearchResult]:
        """Extract results from DuckDuckGo."""
        results = []
        # DDG results are in links with specific patterns
        for link in parsed.links:
            href = link.get("href", "")
            text = link.get("text", "")
            # Skip DDG internal links
            if "duckduckgo.com" in href or not href.startswith("http"):
                continue
            if len(text) > 10:
                results.append(SearchResult(
                    title=text[:100],
                    url=href,
                    snippet="",
                ))
        return results

    def _extract_google_results(self, parsed: ParsedContent) -> List[SearchResult]:
        """Extract results from Google."""
        results = []
        for link in parsed.links:
            href = link.get("href", "")
            text = link.get("text", "")
            if "google.com" in href or not href.startswith("http"):
                continue
            if len(text) > 10:
                results.append(SearchResult(
                    title=text[:100],
                    url=href,
                    snippet="",
                ))
        return results

    def _extract_bing_results(self, parsed: ParsedContent) -> List[SearchResult]:
        """Extract results from Bing (text contains domain + URL + title pattern)."""
        import re
        results = []
        url_pattern = re.compile(r'^https?://[^\s]+$')
        lines = [l.strip() for l in parsed.main_text.split("\n") if l.strip()]

        seen_urls = set()
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for URL on its own line
            if url_pattern.match(line):
                url = line
                # Skip bing/microsoft internal URLs
                if "bing.com" in url or "microsoft.com" in url:
                    i += 1
                    continue
                if url in seen_urls:
                    i += 1
                    continue
                seen_urls.add(url)

                # Title is the line before the URL (if not a URL itself)
                title = ""
                if i > 0 and not url_pattern.match(lines[i - 1]):
                    title = lines[i - 1]

                # Snippet is the line after the URL (if not a URL)
                snippet = ""
                if i + 1 < len(lines) and not url_pattern.match(lines[i + 1]):
                    snippet = lines[i + 1]

                results.append(SearchResult(
                    title=title[:100],
                    url=url,
                    snippet=snippet[:200],
                ))
            i += 1

        return results
