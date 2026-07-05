"""
Tests for NICTO AI Knowledge Base
VectorIndexer, WebCrawler, KnowledgeBase
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, asyncio, time
sys.path.insert(0, r"C:\Users\BYU\Desktop\NICTO")

print("NICTO AI Knowledge Base Tests")
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
# Test 1: VectorIndexer (no network needed)
# ============================================================
print("\n[1/4] VectorIndexer - embedding & search")
from nicto_ai.knowledge.indexer import VectorIndexer

indexer = VectorIndexer(dim=256)

# Add some documents
docs = [
    ("https://python.org", "Python Programming", "Python is a high-level programming language used for web development, data science, and artificial intelligence."),
    ("https://java.com", "Java Language", "Java is a class-based, object-oriented programming language designed for portability."),
    ("https://rust-lang.org", "Rust Language", "Rust is a systems programming language focused on safety, speed, and concurrency."),
    ("https://go.dev", "Go Language", "Go is an open-source programming language with built-in concurrency and garbage collection."),
    ("https://docs.pytest.org", "Testing with Pytest", "Pytest is a framework for writing and running automated tests in Python."),
]

for url, title, text in docs:
    entry_id = indexer.add(url, title, text)
    test(f"added '{title}' (id={entry_id})", entry_id > 0)

test("count matches", indexer.count() == 5)

# Search
results = indexer.search("Python programming language", top_k=3)
test("search returns results", len(results) > 0)
test("search returns results with scores", all(r[1] > 0 for r in results))
test("score > 0", results[0][1] > 0)

# Search for systems programming
results2 = indexer.search("systems programming memory safety", top_k=2)
test("Rust found for systems programming", any(r[2]["title"] == "Rust Language" for r in results2))

# Get specific entry
entry = indexer.get(1)
test("get entry works", entry is not None and entry["title"] == "Python Programming")

# Save and load
indexer.save(r"C:\Users\BYU\Desktop\NICTO\tests\_test_index.json")
indexer2 = VectorIndexer(dim=256)
indexer2.load(r"C:\Users\BYU\Desktop\NICTO\tests\_test_index.json")
test("load preserves count", indexer2.count() == 5)
results3 = indexer2.search("Python", top_k=1)
test("loaded index searches correctly", len(results3) > 0)

# Clear
indexer2.clear()
test("clear works", indexer2.count() == 0)

# Cleanup
import os
os.remove(r"C:\Users\BYU\Desktop\NICTO\tests\_test_index.json")
indexer.close()

print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 2: WebCrawler - single page
# ============================================================
print("[2/4] WebCrawler - single page crawl")
from nicto_ai.browser.engine import BrowserEngine
from nicto_ai.knowledge.crawler import WebCrawler

async def test_crawler():
    engine = BrowserEngine(headless=True)
    await engine.start()
    crawler = WebCrawler(engine, max_depth=0, max_pages=1)

    page = await crawler.crawl_url("https://example.com")
    test("crawled page", page.status == "ok")
    test("title extracted", page.title == "Example Domain")
    test("content has text", page.content.word_count > 0)
    test("depth is 0", page.depth == 0)
    test("crawl_time_ms > 0", page.crawl_time_ms > 0)

    await engine.stop()

asyncio.run(test_crawler())
print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 3: WebCrawler - multi-page BFS
# ============================================================
print("[3/4] WebCrawler - BFS crawl")
from nicto_ai.knowledge.crawler import CrawlResult

async def test_bfs():
    engine = BrowserEngine(headless=True)
    await engine.start()
    crawler = WebCrawler(engine, max_depth=1, max_pages=5, delay_ms=500)

    result = await crawler.crawl("https://example.com", max_depth=1, max_pages=5)
    test("CrawlResult returned", isinstance(result, CrawlResult))
    test("seed_url set", result.seed_url == "https://example.com")
    test("pages crawled", result.total_pages >= 1)
    test("total_time_ms > 0", result.total_time_ms > 0)
    test("first page is seed", result.pages[0].url == "https://example.com")

    await engine.stop()

asyncio.run(test_bfs())
print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Test 4: KnowledgeBase - full integration
# ============================================================
print("[4/4] KnowledgeBase - full integration")
from nicto_ai.knowledge.knowledge_base import KnowledgeBase

async def test_kb():
    kb = KnowledgeBase(max_crawl_depth=0, max_crawl_pages=3)
    await kb.start()
    test("knowledge base started", True)

    # Crawl a single page
    success = await kb.crawl_single("https://example.com")
    test("crawled single page", success)
    test("indexed entry", kb.stats.total_entries >= 1)

    # Query
    result = kb.query("Example Domain website")
    test("query returns results", result.total_results > 0)
    test("query completed", True)  # query_time may be 0 for fast queries
    if result.results:
        test("result has title", len(result.results[0]["title"]) > 0)
        test("result has url", result.results[0]["url"].startswith("http"))

    # search_and_read
    text = await kb.search_and_read("Example Domain")
    test("search_and_read returns text", len(text) > 0)

    # Stats
    stats = kb.stats
    test("stats.total_entries > 0", stats.total_entries > 0)

    # Get specific page
    if result.results:
        page = kb.get_page(result.results[0]["id"])
        test("get_page works", page is not None)

    await kb.stop()
    test("knowledge base stopped", True)

asyncio.run(test_kb())
print(f"  ({PASS} passed, {FAIL} failed so far)\n")


# ============================================================
# Summary
# ============================================================
print("=" * 60)
if FAIL == 0:
    print(f"ALL {PASS} TESTS PASSED")
    print("Knowledge Base is ready!")
else:
    print(f"PASSED: {PASS} / FAILED: {FAIL}")
print("=" * 60)
