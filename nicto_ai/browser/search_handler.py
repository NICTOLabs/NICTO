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

    Uses HTTP requests (via requests library or BrowserEngine)
    to perform searches and parse results.
    """

    def __init__(self, browser: BrowserEngine, parser: Optional[PageParser] = None):
        self.browser = browser
        self.parser = parser or PageParser()

    def _http_search(self, url: str, engine: str) -> str:
        """Fetch a search URL via HTTP with TLS fingerprint impersonation."""
        from curl_cffi import requests as curl_requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
        }

        resp = curl_requests.get(url, headers=headers, impersonate="chrome131", timeout=15)
        resp.raise_for_status()
        return resp.text

    async def search(
        self,
        query: str,
        engine: str = "duckduckgo",
        max_results: int = 10,
        fetch_content: bool = False,
    ) -> SearchResponse:
        """
        Perform a web search and extract results.

        Uses HTTP requests for search engines that support it (ddg lite, bing),
        falls back to headless browser for others.

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

        # Build search URLs
        urls = {
            "bing": f"https://www.bing.com/search?q={query.replace(' ', '+')}&count={max_results}",
            "duckduckgo": f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}",
        }

        results = []

        # Try HTTP-based search first (uses curl_cffi for TLS fingerprint impersonation)
        engines_to_try = [engine] if engine in urls else ["bing", "duckduckgo"]
        for eng in engines_to_try:
            url = urls.get(eng)
            if not url:
                continue
            try:
                html = await asyncio.get_event_loop().run_in_executor(
                    None, self._http_search, url, eng
                )
                parsed = self.parser.parse_html(html, url)
                # Store raw HTML for extractors that need it (e.g., Bing)
                parsed.metadata["_raw_html"] = html
                extracted = self._extract_search_results(parsed, eng)
                if extracted:
                    results = extracted
                    engine = eng
                    break
            except Exception as e:
                logger.debug("HTTP search on %s failed: %s", eng, e)

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
        """Extract results from DuckDuckGo (Lite version)."""
        from bs4 import BeautifulSoup
        results = []
        seen_urls = set()

        # For DDG Lite, results are in anchor tags with rel="nofollow"
        for link in parsed.links:
            href = link.get("href", "")
            text = link.get("text", "").strip()
            # Skip DDG internal links and non-http links
            if "duckduckgo.com" in href or not href.startswith("http"):
                continue
            if href in seen_urls:
                continue
            if not text or len(text) < 2:
                continue
            seen_urls.add(href)
            results.append(SearchResult(
                title=text[:120],
                url=href,
                snippet="",
            ))

        # Deduplicate by URL
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
        """Extract results from Bing using BeautifulSoup to find actual result URLs."""
        from bs4 import BeautifulSoup
        import re

        results = []
        seen_urls = set()

        # Re-parse from raw HTML for proper element structure
        raw_html = ""
        if isinstance(parsed.metadata, dict):
            raw_html = parsed.metadata.get("_raw_html", "")
        soup = BeautifulSoup(raw_html, "html.parser") if raw_html else None

        # Find tilk-class links (these contain the embedded real URL in their text)
        for a_tag in (soup.find_all("a", class_="tilk") if soup else []):
            text = a_tag.get_text(strip=True)
            # Real URL is embedded in text: "domainhttps://real.url/path"
            url_match = re.search(r'https?://[^\s\u00A0<>]+', text)
            if not url_match:
                continue
            url = url_match.group()
            if url in seen_urls or "bing.com" in url or "microsoft.com" in url:
                continue
            seen_urls.add(url)

            # Title comes from h2 parent
            h2 = a_tag.find_parent("h2")
            title = h2.get_text(strip=True) if h2 else text.split("https://")[0].strip()
            # Clean title
            title = re.sub(r'\s+', ' ', title).strip()[:120]

            results.append(SearchResult(title=title, url=url, snippet=""))

        if not results and parsed.links:
            # Fallback: find URLs embedded in link text
            for link in parsed.links:
                text = link.get("text", "").strip()
                url_match = re.search(r'https?://[^\s\u00A0<>]+', text)
                if url_match:
                    url = url_match.group()
                    if url in seen_urls or "bing.com" in url or "microsoft.com" in url:
                        continue
                    seen_urls.add(url)
                    title = text.split("https://")[0].strip()
                    title = re.sub(r'\s+', ' ', title).strip()[:120]
                    results.append(SearchResult(title=title or url, url=url, snippet=""))

        return results
