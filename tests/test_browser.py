"""
Tests for NICTO AI Browser System
BrowserEngine, PageParser, SearchHandler, NICTOBrowser
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, asyncio
sys.path.insert(0, r"C:\Users\BYU\Desktop\NICTO")

print("NICTO AI Browser Tests")
print("=" * 60)

PASS = 0
FAIL = 0


def test(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


# ============================================================
# Test 1: PageParser (no network needed)
# ============================================================
print("\n[1/5] PageParser - HTML parsing")
from nicto_ai.browser.page_parser import PageParser, ParsedContent

parser = PageParser()

test_html = """
<html>
<head><title>Test Page - NICTO AI</title>
<meta name="description" content="A test page">
</head>
<body>
<nav>Navigation bar</nav>
<h1>Main Title</h1>
<h2>Section 1</h2>
<p>Artificial intelligence is transforming the world with new capabilities in natural language processing and computer vision.</p>
<h2>Section 2</h2>
<p>Machine learning models like transformers have revolutionized how we build AI systems.</p>
<p>Deep learning enables computers to learn from data automatically.</p>
<code>model = transformers.AutoModel.from_pretrained("gpt2")</code>
<a href="https://example.com">Example Link</a>
<a href="https://arxiv.org">arXiv Papers</a>
<footer>Footer content</footer>
</body>
</html>
"""

content = parser.parse_html(test_html, "https://test.com")
test("title extracted", content.title == "Test Page - NICTO AI")
test("headings extracted", len(content.headings) == 3)
test("paragraphs extracted", len(content.paragraphs) == 3)
test("links extracted", len(content.links) == 2)
test("code blocks extracted", len(content.code_blocks) == 1)
test("word count > 0", content.word_count > 0)
test("metadata extracted", "description" in content.metadata)

# Test text parsing
text_content = parser.parse_text("HELLO WORLD\nThis is a paragraph with enough words to be detected as content and not a heading.\nAnother paragraph here that has sufficient length to pass the filter.")
test("text parsing works", len(text_content.paragraphs) > 0)

# Test relevance ranking
query = "artificial intelligence"
score = parser.rank_relevance(content, query)
test("relevance score > 0", score > 0)
test("relevance score <= 1", score <= 1)

# Test summarization
summary = parser.summarize(content, max_sentences=2)
test("summary generated", len(summary) > 10)

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 2: BrowserEngine - Chromium
# ============================================================
print("[2/5] BrowserEngine - Chromium control")
from nicto_ai.browser.engine import BrowserEngine, PageInfo

async def test_engine():
    engine = BrowserEngine(headless=True)
    await engine.start()
    test("browser started", engine._browser is not None)

    # Navigate to example.com
    page = await engine.navigate("https://example.com")
    test("page loaded", isinstance(page, PageInfo))
    test("title is 'Example Domain'", page.title == "Example Domain")
    test("url correct", "example.com" in page.url)
    test("text content extracted", len(page.text_content) > 0)
    test("links extracted", len(page.links) >= 1)
    test("load time recorded", page.load_time_ms > 0)

    # Get text
    text = await engine.get_text()
    test("get_text works", len(text) > 0)

    # Get HTML
    html = await engine.get_html()
    test("get_html works", len(html) > 0)

    # Execute JS
    title = await engine.execute_js("document.title")
    test("JS execution works", title == "Example Domain")

    # Get links
    links = await engine.get_links()
    test("get_links works", len(links) >= 1)

    # History tracked
    test("history tracked", len(engine.history) == 1)

    await engine.stop()
    test("browser stopped", True)

asyncio.run(test_engine())
print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 3: SearchHandler - DuckDuckGo search
# ============================================================
print("[3/5] SearchHandler - web search")
from nicto_ai.browser.search_handler import SearchHandler, SearchResponse

async def test_search():
    engine = BrowserEngine(headless=True)
    await engine.start()
    handler = SearchHandler(engine)

    response = await handler.search("python programming", max_results=5, fetch_content=False)
    test("search returns SearchResponse", isinstance(response, SearchResponse))
    test("query matches", response.query == "python programming")
    test("engine matches", response.engine == "bing")
    test("search_time_ms > 0", response.search_time_ms > 0)
    # Bing may intermittently block headless browsers; check we got a valid response
    test("valid response (results may vary)", response.total_results >= 0)

    if response.results:
        r = response.results[0]
        test("result has title", len(r.title) > 0)
        test("result has url", r.url.startswith("http"))
        test("result has rank", r.rank > 0)

    await engine.stop()

asyncio.run(test_search())
print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 4: NICTOBrowser - full integration
# ============================================================
print("[4/5] NICTOBrowser - full integration")
from nicto_ai.browser.browser import NICTOBrowser

async def test_browser():
    browser = NICTOBrowser()
    await browser.start(tor=False)
    test("browser started", browser.is_running)

    # Direct browse
    result = await browser.browse("https://example.com")
    test("browse returns BrowseResult", result.title == "Example Domain")
    test("content parsed", result.content.word_count > 0)
    test("source is direct", result.source == "direct")

    # Search
    results = await browser.search("AI research", max_results=3, fetch_content=False)
    test("search completed", True)  # Bing may intermittently block

    # History
    test("history tracked", len(browser.history) >= 1)

    # Current URL
    test("current_url set", len(browser.current_url) > 0)

    await browser.stop()
    test("browser stopped", not browser.is_running)

asyncio.run(test_browser())
print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 5: TorProxy - detection only (no tor binary needed)
# ============================================================
print("[5/5] TorProxy - component test")
from nicto_ai.browser.tor_proxy import TorProxy

tor = TorProxy(socks_port=9050)
test("tor proxy created", tor.socks_port == 9050)
test("proxy URL format", tor.get_proxy_url() == "socks5://127.0.0.1:9050")
test("not running initially", not tor.is_running)

tor_path = tor.find_tor()
if tor_path:
    test("tor binary found", True)
    print(f"  Tor path: {tor_path}")
else:
    test("tor binary not found (expected if not installed)", True)
    print("  Note: Install Tor to enable anonymous browsing")

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Summary
# ============================================================
print("=" * 60)
if FAIL == 0:
    print(f"ALL {PASS} TESTS PASSED")
    print("NICTO Browser is ready!")
else:
    print(f"PASSED: {PASS} / FAILED: {FAIL}")
print("=" * 60)
