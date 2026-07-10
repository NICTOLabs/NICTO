"""
NICTO AI - Browser Engine
Headless Chromium control via Playwright for autonomous web browsing
"""

import asyncio
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PageInfo:
    """Structured info about a loaded page"""
    url: str
    title: str
    text_content: str
    html: str
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    status_code: int = 200
    load_time_ms: float = 0.0


class BrowserEngine:
    """
    Headless Chromium browser engine for NICTO AI.

    Manages a Playwright Chromium instance with:
    - Headless browsing (no visible UI)
    - Page navigation, content extraction
    - JavaScript execution
    - Screenshot capture
    - Cookie/session management
    - Optional Tor proxy routing
    """

    def __init__(
        self,
        headless: bool = True,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout_ms: int = 30000,
    ):
        self.headless = headless
        self.proxy = proxy
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout_ms = timeout_ms

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._history: List[str] = []

    async def start(self):
        """Launch the browser."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        launch_args = {
            "headless": self.headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        }
        if self.proxy:
            launch_args["proxy"] = {"server": self.proxy}

        self._browser = await self._playwright.chromium.launch(**launch_args)

        context_args = {
            "viewport": {"width": self.viewport_width, "height": self.viewport_height},
            "user_agent": self.user_agent,
        }
        self._context = await self._browser.new_context(**context_args)
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.timeout_ms)

        # Apply stealth patches to avoid bot detection
        try:
            from playwright_stealth import stealth_async
            await stealth_async(self._page)
        except ImportError:
            # Fallback: basic anti-detection
            await self._page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                window.chrome = {runtime: {}};
            """)

        logger.info("Browser engine started (headless=%s, proxy=%s)", self.headless, self.proxy)

    async def stop(self):
        """Close the browser."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser engine stopped")

    async def navigate(self, url: str) -> PageInfo:
        """
        Navigate to a URL and return page info.

        Args:
            url: Target URL

        Returns:
            PageInfo with extracted content
        """
        import time
        start = time.time()

        # Navigate to blank first to ensure clean page state
        try:
            current = self._page.url
            if current not in ("about:blank", ""):
                await self._page.goto("about:blank", wait_until="domcontentloaded", timeout=3000)
        except Exception:
            pass

        response = await self._page.goto(url, wait_until="domcontentloaded")
        load_time = (time.time() - start) * 1000

        # Wait a bit for dynamic content (non-fatal if timeout)
        try:
            await self._page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass  # Continue even if networkidle times out

        self._history.append(url)

        return await self._extract_page_info(response.status if response else 200, load_time)

    async def search(self, query: str, engine: str = "duckduckgo") -> PageInfo:
        """
        Perform a search using a search engine.

        Args:
            query: Search query
            engine: Search engine to use ("duckduckgo", "google", "bing")

        Returns:
            PageInfo with search results
        """
        urls = {
            "duckduckgo": f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}",
            "google": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "bing": f"https://www.bing.com/search?q={query.replace(' ', '+')}",
        }
        url = urls.get(engine, urls["duckduckgo"])
        return await self.navigate(url)

    async def get_text(self) -> str:
        """Get the full text content of the current page."""
        return await self._page.inner_text("body")

    async def get_html(self) -> str:
        """Get the full HTML of the current page."""
        return await self._page.content()

    async def execute_js(self, script: str):
        """Execute JavaScript on the current page."""
        return await self._page.evaluate(script)

    async def screenshot(self, path: Optional[str] = None) -> bytes:
        """Take a screenshot of the current page."""
        return await self._page.screenshot(path=path)

    async def get_links(self) -> List[Dict[str, str]]:
        """Extract all links from the current page."""
        links = await self._page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: a.innerText.trim(),
                href: a.href,
            })).filter(l => l.text.length > 0)
        """)
        return links

    async def click(self, selector: str):
        """Click an element by CSS selector."""
        await self._page.click(selector)

    async def fill(self, selector: str, value: str):
        """Fill a form field."""
        await self._page.fill(selector, value)

    async def scroll_down(self, pixels: int = 500):
        """Scroll the page down."""
        await self._page.evaluate(f"window.scrollBy(0, {pixels})")

    async def get_cookies(self) -> List[Dict]:
        """Get all cookies for the current context."""
        return await self._context.cookies()

    async def clear_cookies(self):
        """Clear all cookies."""
        await self._context.clear_cookies()

    async def new_page(self):
        """Create a new tab in the same browser context and return the page object."""
        page = await self._context.new_page()
        page.set_default_timeout(self.timeout_ms)
        return page

    async def search_on_fresh_page(self, url: str) -> PageInfo:
        """Navigate to a URL on a fresh page, then restore the main page."""
        import time
        start = time.time()

        fresh_page = await self._context.new_page()
        fresh_page.set_default_timeout(self.timeout_ms)

        # Apply stealth to fresh page too
        try:
            from playwright_stealth import stealth_async
            await stealth_async(fresh_page)
        except ImportError:
            pass

        try:
            response = await fresh_page.goto(url, wait_until="domcontentloaded")
            try:
                await fresh_page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            # Extract info from fresh page
            title = await fresh_page.title()
            text = await fresh_page.inner_text("body")
            html = await fresh_page.content()

            links = await fresh_page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    text: a.innerText.trim(),
                    href: a.href,
                })).filter(l => l.text.length > 0)
            """)

            load_time = (time.time() - start) * 1000

            return PageInfo(
                url=fresh_page.url,
                title=title,
                text_content=text,
                html=html,
                links=links,
                status_code=response.status if response else 200,
                load_time_ms=load_time,
            )
        finally:
            await fresh_page.close()

    async def _extract_page_info(self, status_code: int, load_time: float) -> PageInfo:
        """Extract structured info from the current page."""
        title = await self._page.title()
        text = await self._page.inner_text("body")
        html = await self._page.content()
        links = await self.get_links()

        images = await self._page.evaluate("""
            () => Array.from(document.querySelectorAll('img[src]')).map(img => ({
                src: img.src,
                alt: img.alt || '',
            }))
        """)

        return PageInfo(
            url=self._page.url,
            title=title,
            text_content=text,
            html=html,
            links=links,
            images=images,
            status_code=status_code,
            load_time_ms=load_time,
        )

    @property
    def current_url(self) -> str:
        return self._page.url if self._page else ""

    @property
    def history(self) -> List[str]:
        return list(self._history)
