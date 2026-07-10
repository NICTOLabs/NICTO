"""
NICTO AI - Knowledge Base
Crawls, indexes, and retrieves knowledge via HTTP (no browser needed)
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

from ..browser.page_parser import PageParser
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
    total_entries: int = 0
    last_action: str = ""
    index_path: str = ""


class KnowledgeBase:
    """
    NICTO's knowledge base - crawls the web via HTTP and builds searchable knowledge.

    Uses curl_cffi for reliable HTTP requests (no browser needed).
    VectorIndexer provides TF-IDF semantic search over crawled content.

    Usage:
        kb = KnowledgeBase()
        kb.start_sync()

        # Crawl a URL
        kb.crawl_sync("https://docs.python.org/3/")

        # Add text directly
        kb.ingest_text("Python is a programming language", url="manual", title="Python")

        # Search
        results = kb.query("How do I create a list?")

        kb.stop_sync()
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
    ):
        self.parser = PageParser()
        self.indexer: Optional[VectorIndexer] = None
        self.index_path = index_path

    def start_sync(self):
        """Start the knowledge base (no browser needed)."""
        self.indexer = VectorIndexer(dim=256, db_path=self.index_path)
        logger.info("Knowledge base started (index: %s)", self.index_path or ":memory:")

    def stop_sync(self):
        """Stop the knowledge base."""
        if self.indexer:
            self.indexer.close()
        logger.info("Knowledge base stopped")

    def crawl_sync(
        self,
        url: str,
        max_depth: int = 1,
        max_pages: int = 10,
    ):
        """
        Crawl a URL and index its content via HTTP.

        Args:
            url: Starting URL
            max_depth: How many levels of links to follow (0 = single page)
            max_pages: Maximum pages to crawl

        Returns:
            Dict with crawl statistics
        """
        if self.indexer is None:
            raise RuntimeError("Knowledge base not started. Call start_sync() first.")

        from curl_cffi import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.5",
        }

        visited = set()
        queue = [(url, 0)]
        indexed = 0
        errors = 0
        start_time = time.time()

        while queue and indexed < max_pages:
            page_url, depth = queue.pop(0)
            if page_url in visited or depth > max_depth:
                continue
            visited.add(page_url)

            try:
                resp = requests.get(page_url, headers=headers, impersonate="chrome131", timeout=10)
                if resp.status_code != 200:
                    errors += 1
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else urlparse(page_url).path.split("/")[-1]

                # Extract main text
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                main_text = soup.get_text(separator="\n", strip=True)
                main_text = re.sub(r'\n+', '\n', main_text).strip()

                if len(main_text) > 20:
                    summary = main_text[:300].replace("\n", " ").strip()
                    self.indexer.add(
                        url=page_url,
                        title=title,
                        text=main_text[:10000],
                        summary=summary[:1000],
                        metadata={"depth": depth, "status": resp.status_code},
                    )
                    indexed += 1

                # Extract links for further crawling
                if depth < max_depth:
                    base_domain = urlparse(url).netloc
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        full_url = urljoin(page_url, href)
                        parsed = urlparse(full_url)
                        if parsed.netloc == base_domain and full_url not in visited and full_url.startswith("http"):
                            queue.append((full_url, depth + 1))

            except Exception as e:
                logger.debug("Failed to crawl %s: %s", page_url, e)
                errors += 1

        elapsed = (time.time() - start_time) * 1000
        logger.info("Crawled %d pages from %s in %.1fs (%d errors)", indexed, url, elapsed / 1000, errors)

        return {
            "url": url,
            "pages_crawled": indexed,
            "errors": errors,
            "time_ms": elapsed,
            "total_in_index": self.indexer.count(),
        }

    def ingest_text(self, text: str, title: str = "", url: str = "", summary: str = ""):
        """
        Add text directly to the knowledge base.

        Args:
            text: Content text
            title: Optional title
            url: Optional source URL
            summary: Optional summary
        """
        if self.indexer is None:
            raise RuntimeError("Knowledge base not started.")
        if not text.strip():
            return
        if not summary:
            summary = text[:300].replace("\n", " ").strip()
        self.indexer.add(
            url=url,
            title=title or "untitled",
            text=text[:10000],
            summary=summary[:1000],
        )

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
        if self.indexer is None:
            raise RuntimeError("Knowledge base not started.")

        start = time.time()
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
                    "text": r[2].get("text", "")[:500] if "text" in r[2] else r[2].get("summary", ""),
                }
                for r in results
            ],
            total_results=len(results),
            query_time_ms=elapsed,
        )

    @property
    def stats(self) -> KnowledgeStats:
        """Get knowledge base statistics."""
        count = self.indexer.count() if self.indexer else 0
        return KnowledgeStats(total_entries=count, index_path=str(self.index_path or ":memory:"))
