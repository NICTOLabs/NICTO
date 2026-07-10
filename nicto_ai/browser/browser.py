"""
NICTO AI - Main Browser Orchestrator
Ties together Chromium engine, Tor proxy, parser, and search handler
"""

import asyncio
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from .engine import BrowserEngine, PageInfo
from .tor_proxy import TorProxy
from .page_parser import PageParser, ParsedContent
from .search_handler import SearchHandler, SearchResponse, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class BrowseResult:
    """Result from a browsing operation"""
    url: str
    title: str
    content: ParsedContent
    search_query: Optional[str] = None
    source: str = "direct"  # "direct", "search", "deep_search"


class NICTOBrowser:
    """
    NICTO AI's internal web browser.

    This is NOT a user-facing browser. It's NICTO's autonomous
    information gathering system that can:
    - Navigate to any URL
    - Search the web via multiple engines
    - Extract and parse content from pages
    - Browse anonymously via Tor
    - Follow links and build knowledge
    - Take screenshots for visual understanding

    Usage:
        browser = NICTOBrowser()
        await browser.start(tor=True)

        # Direct browsing
        result = await browser.browse("https://en.wikipedia.org/wiki/Artificial_intelligence")

        # Web search
        results = await browser.search("latest AI research 2026")

        # Deep search (search + follow links)
        deep = await browser.deep_search("transformer architecture explained", depth=2)

        await browser.stop()
    """

    def __init__(self):
        self.engine: Optional[BrowserEngine] = None
        self.tor: Optional[TorProxy] = None
        self.parser: PageParser = PageParser()
        self.search_handler: Optional[SearchHandler] = None
        self._browse_history: List[BrowseResult] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(
        self,
        tor: bool = False,
        headless: bool = True,
        proxy: Optional[str] = None,
    ):
        """
        Start the browser.

        Args:
            tor: Enable Tor proxy for anonymous browsing
            headless: Run browser in headless mode
            proxy: Custom proxy URL (overrides tor)
        """
        # Determine proxy
        actual_proxy = proxy
        if tor and not proxy:
            self.tor = TorProxy()
            if self.tor.find_tor():
                tor_started = await asyncio.get_event_loop().run_in_executor(
                    None, self.tor.start
                )
                if tor_started:
                    actual_proxy = self.tor.get_proxy_url()
                    logger.info("Using Tor proxy: %s", actual_proxy)
                else:
                    logger.warning("Tor failed to start, falling back to direct connection")
                    self.tor = None
            else:
                logger.warning("Tor not found, falling back to direct connection")
                self.tor = None

        # Start browser engine
        self.engine = BrowserEngine(
            headless=headless,
            proxy=actual_proxy,
        )
        await self.engine.start()

        # Create search handler
        self.search_handler = SearchHandler(self.engine, self.parser)

        logger.info("NICTO Browser started (tor=%s, proxy=%s)", tor, actual_proxy)

    async def stop(self):
        """Stop the browser and all components."""
        if self.engine:
            await self.engine.stop()
        if self.tor:
            self.tor.stop()
        logger.info("NICTO Browser stopped")

    async def browse(self, url: str) -> BrowseResult:
        """
        Navigate to a URL and extract content.

        Args:
            url: Target URL

        Returns:
            BrowseResult with parsed content
        """
        page = await self.engine.navigate(url)
        content = self.parser.parse_html(page.html, url)
        summary = self.parser.summarize(content)

        result = BrowseResult(
            url=url,
            title=content.title or page.title,
            content=content,
            source="direct",
        )
        content.summary = summary
        self._browse_history.append(result)
        return result

    async def search(
        self,
        query: str,
        engine: str = "duckduckgo",
        max_results: int = 10,
        fetch_content: bool = True,
    ) -> List[SearchResult]:
        """
        Search the web.

        Args:
            query: Search query
            engine: Search engine to use
            max_results: Maximum results
            fetch_content: Visit each result URL for full content

        Returns:
            List of SearchResult objects
        """
        response = await self.search_handler.search(
            query, engine, max_results, fetch_content
        )

        # Add to browse history
        for result in response.results:
            self._browse_history.append(BrowseResult(
                url=result.url,
                title=result.title,
                content=result.content or ParsedContent(title=result.title, main_text=result.snippet),
                search_query=query,
                source="search",
            ))

        return response.results

    # ── Synchronous wrappers for tool use ──────────────────────────

    @property
    def _event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create a persistent event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def start_sync(
        self,
        tor: bool = False,
        headless: bool = True,
        proxy: str = None,
    ):
        """Synchronous wrapper around start()."""
        loop = self._event_loop
        return loop.run_until_complete(self.start(tor=tor, headless=headless, proxy=proxy))

    def stop_sync(self):
        """Synchronous wrapper around stop()."""
        try:
            loop = self._event_loop
            result = loop.run_until_complete(self.stop())
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            self._loop = None
            return result
        except Exception:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            self._loop = None

    def search_sync(
        self,
        query: str,
        engine: str = "bing",
        max_results: int = 10,
        fetch_content: bool = True,
    ) -> list:
        """Synchronous wrapper around search()."""
        loop = self._event_loop
        return loop.run_until_complete(
            self.search(query, engine=engine, max_results=max_results, fetch_content=fetch_content)
        )

    def browse_sync(self, url: str) -> BrowseResult:
        """Synchronous wrapper around browse()."""
        loop = self._event_loop
        return loop.run_until_complete(self.browse(url))

    async def deep_search(
        self,
        query: str,
        depth: int = 2,
        max_results: int = 5,
    ) -> SearchResponse:
        """
        Deep search: search + follow links for richer information.

        Args:
            query: Search query
            depth: How many levels of link-following
            max_results: Results per level

        Returns:
            SearchResponse with deep results
        """
        return await self.search_handler.deep_search(query, depth, max_results)

    async def new_identity(self):
        """Request a new Tor identity (new IP)."""
        if self.tor:
            await asyncio.get_event_loop().run_in_executor(
                None, self.tor.new_identity
            )
            logger.info("New Tor identity requested")

    async def screenshot(self, path: Optional[str] = None) -> bytes:
        """Take a screenshot of the current page."""
        return await self.engine.screenshot(path)

    async def get_links(self) -> List[Dict[str, str]]:
        """Get all links on the current page."""
        return await self.engine.get_links()

    @property
    def current_url(self) -> str:
        return self.engine.current_url if self.engine else ""

    @property
    def history(self) -> List[BrowseResult]:
        return list(self._browse_history)

    @property
    def is_running(self) -> bool:
        if self.engine is None or self.engine._browser is None:
            return False
        try:
            return self.engine._browser.is_connected()
        except Exception:
            return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        asyncio.get_event_loop().run_until_complete(self.stop())
