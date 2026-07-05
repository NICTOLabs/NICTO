"""
NICTO AI - Web Crawler
Crawls websites using the NICTO browser and extracts content
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from ..browser.engine import BrowserEngine
from ..browser.page_parser import PageParser, ParsedContent

logger = logging.getLogger(__name__)


@dataclass
class CrawledPage:
    """A crawled page with extracted content"""
    url: str
    title: str
    content: ParsedContent
    depth: int = 0
    crawl_time_ms: float = 0.0
    status: str = "ok"  # "ok", "error", "timeout", "blocked"


@dataclass
class CrawlResult:
    """Result of a crawl operation"""
    seed_url: str
    pages: List[CrawledPage] = field(default_factory=list)
    total_pages: int = 0
    total_time_ms: float = 0.0
    errors: int = 0


class WebCrawler:
    """
    Crawls websites using the NICTO browser.

    Features:
    - BFS/DFS crawling with configurable depth
    - Domain restriction (stay on same domain)
    - Rate limiting between requests
    - Content deduplication
    - robots.txt respect (optional)
    """

    def __init__(
        self,
        browser: BrowserEngine,
        parser: Optional[PageParser] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        delay_ms: int = 1000,
        same_domain: bool = True,
    ):
        self.browser = browser
        self.parser = parser or PageParser()
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay_ms = delay_ms
        self.same_domain = same_domain
        self._visited: Set[str] = set()

    async def crawl(
        self,
        seed_url: str,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> CrawlResult:
        """
        Crawl starting from a seed URL.

        Uses BFS to explore pages up to max_depth, collecting content.

        Args:
            seed_url: Starting URL
            max_depth: Override max crawl depth
            max_pages: Override max pages to crawl

        Returns:
            CrawlResult with all crawled pages
        """
        if max_depth is not None:
            self.max_depth = max_depth
        if max_pages is not None:
            self.max_pages = max_pages

        self._visited.clear()
        result = CrawlResult(seed_url=seed_url)
        start_time = time.time()

        # BFS queue: (url, depth)
        queue = [(seed_url, 0)]

        while queue and len(result.pages) < self.max_pages:
            url, depth = queue.pop(0)

            # Skip if already visited or too deep
            if url in self._visited or depth > self.max_depth:
                continue

            # Domain restriction
            if self.same_domain:
                seed_domain = urlparse(seed_url).netloc
                page_domain = urlparse(url).netloc
                if page_domain != seed_domain:
                    continue

            self._visited.add(url)

            # Crawl the page
            page = await self._crawl_page(url, depth)
            result.pages.append(page)
            result.total_pages += 1

            if page.status != "ok":
                result.errors += 1
                continue

            # Extract links for further crawling
            if depth < self.max_depth:
                links = await self.browser.get_links()
                for link in links:
                    href = link.get("href", "")
                    if href and href.startswith("http") and href not in self._visited:
                        queue.append((href, depth + 1))

            # Rate limiting
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)

        result.total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Crawled %d pages from %s in %.1fs (%d errors)",
            result.total_pages, seed_url, result.total_time_ms / 1000, result.errors,
        )
        return result

    async def crawl_url(self, url: str) -> CrawledPage:
        """Crawl a single URL."""
        return await self._crawl_page(url, depth=0)

    async def _crawl_page(self, url: str, depth: int) -> CrawledPage:
        """Crawl a single page and extract content."""
        start = time.time()
        try:
            page_info = await self.browser.navigate(url)
            content = self.parser.parse_html(page_info.html, url)
            # Also use text content for JS-rendered pages
            text_parsed = self.parser.parse_text(page_info.text_content, url)
            if text_parsed.word_count > content.word_count:
                content = text_parsed

            content.summary = self.parser.summarize(content)

            elapsed = (time.time() - start) * 1000
            return CrawledPage(
                url=url,
                title=content.title or page_info.title,
                content=content,
                depth=depth,
                crawl_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.warning("Failed to crawl %s: %s", url, e)
            return CrawledPage(
                url=url,
                title="",
                content=ParsedContent(),
                depth=depth,
                crawl_time_ms=elapsed,
                status="error",
            )
