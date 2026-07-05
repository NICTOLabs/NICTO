"""
NICTO AI - Knowledge Base
Orchestrates crawling, indexing, and retrieval of web knowledge
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..browser.engine import BrowserEngine
from ..browser.page_parser import PageParser
from ..browser.browser import NICTOBrowser
from .crawler import WebCrawler, CrawledPage, CrawlResult
from .indexer import VectorIndexer

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeQuery:
    """Result from a knowledge base query"""
    query: str
    results: List[Dict] = field(default_factory=list)
    total_results: int = 0
    query_time_ms: float = 0.0


@dataclass
class KnowledgeStats:
    """Statistics about the knowledge base"""
    total_pages: int = 0
    total_entries: int = 0
    domains: List[str] = field(default_factory=list)
    last_crawl_url: str = ""
    last_crawl_time_ms: float = 0.0


class KnowledgeBase:
    """
    NICTO's knowledge base - crawls the web and builds searchable knowledge.

    Combines the browser (for crawling) with the vector indexer (for retrieval).
    NICTO can use this to:
    1. Learn from any website
    2. Build persistent knowledge
    3. Answer questions from crawled data
    4. Track and update information over time

    Usage:
        kb = KnowledgeBase()
        await kb.start()

        # Crawl a website
        await kb.crawl("https://docs.python.org/3/")

        # Search knowledge
        results = kb.query("How do I create a list in Python?")

        # Get stats
        stats = kb.stats()

        await kb.stop()
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        max_crawl_depth: int = 2,
        max_crawl_pages: int = 50,
    ):
        self.browser = NICTOBrowser()
        self.parser = PageParser()
        self.indexer: Optional[VectorIndexer] = None
        self.crawler: Optional[WebCrawler] = None
        self.index_path = index_path
        self.max_crawl_depth = max_crawl_depth
        self.max_crawl_pages = max_crawl_pages

    async def start(self, tor: bool = False):
        """Start the knowledge base."""
        await self.browser.start(tor=tor)
        self.indexer = VectorIndexer(
            dim=256,
            db_path=self.index_path,
        )
        self.crawler = WebCrawler(
            browser=self.browser.engine,
            parser=self.parser,
            max_depth=self.max_crawl_depth,
            max_pages=self.max_crawl_pages,
        )
        logger.info("Knowledge base started")

    async def stop(self):
        """Stop the knowledge base."""
        await self.browser.stop()
        if self.indexer:
            self.indexer.close()
        logger.info("Knowledge base stopped")

    async def crawl(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> CrawlResult:
        """
        Crawl a website and index its content.

        Args:
            url: Starting URL
            max_depth: Override crawl depth
            max_pages: Override max pages

        Returns:
            CrawlResult with crawl statistics
        """
        if self.crawler is None:
            raise RuntimeError("Knowledge base not started. Call start() first.")

        result = await self.crawler.crawl(url, max_depth, max_pages)

        # Index all crawled pages
        for page in result.pages:
            if page.status == "ok" and page.content.word_count > 5:
                self.indexer.add(
                    url=page.url,
                    title=page.title,
                    text=page.content.main_text,
                    summary=page.content.summary,
                    metadata={
                        "depth": page.depth,
                        "word_count": page.content.word_count,
                        "crawl_time_ms": page.crawl_time_ms,
                    },
                )

        logger.info(
            "Indexed %d pages from %s (total in index: %d)",
            len(result.pages), url, self.indexer.count(),
        )
        return result

    async def crawl_single(self, url: str) -> bool:
        """
        Crawl and index a single page.

        Args:
            url: URL to crawl

        Returns:
            True if successful
        """
        if self.crawler is None or self.indexer is None:
            raise RuntimeError("Knowledge base not started.")

        page = await self.crawler.crawl_url(url)
        if page.status == "ok" and page.content.word_count > 5:
            self.indexer.add(
                url=page.url,
                title=page.title,
                text=page.content.main_text,
                summary=page.content.summary,
            )
            return True
        return False

    def query(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> KnowledgeQuery:
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            KnowledgeQuery with results
        """
        import time
        start = time.time()

        if self.indexer is None:
            raise RuntimeError("Knowledge base not started.")

        results = self.indexer.search(query, top_k, min_score)
        elapsed = (time.time() - start) * 1000

        return KnowledgeQuery(
            query=query,
            results=[
                {
                    "id": r[0],
                    "score": r[1],
                    "url": r[2].get("url", ""),
                    "title": r[2].get("title", ""),
                    "summary": r[2].get("summary", ""),
                }
                for r in results
            ],
            total_results=len(results),
            query_time_ms=elapsed,
        )

    def get_page(self, entry_id: int) -> Optional[Dict]:
        """Get a specific page by entry ID."""
        if self.indexer is None:
            return None
        return self.indexer.get(entry_id)

    @property
    def stats(self) -> KnowledgeStats:
        """Get knowledge base statistics."""
        if self.indexer is None:
            return KnowledgeStats()
        return KnowledgeStats(
            total_entries=self.indexer.count(),
        )

    async def search_and_read(
        self,
        query: str,
        max_results: int = 3,
    ) -> str:
        """
        Search knowledge base and return formatted results as text.

        Useful for feeding into NICTO's reasoning.

        Args:
            query: Search query
            max_results: Max results to include

        Returns:
            Formatted text with search results
        """
        results = self.query(query, top_k=max_results)
        if not results.results:
            return f"No knowledge found for: {query}"

        lines = [f"Knowledge results for: {query}\n"]
        for i, r in enumerate(results.results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            lines.append(f"    {r['summary']}")
            lines.append(f"    Score: {r['score']:.3f}")
            lines.append("")

        return "\n".join(lines)
