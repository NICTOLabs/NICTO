"""
NICTO AI - Page Parser
Extracts structured content from web pages
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ParsedContent:
    """Structured content extracted from a web page"""
    title: str = ""
    main_text: str = ""
    summary: str = ""
    headings: List[Dict[str, str]] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=list)
    word_count: int = 0
    relevance_score: float = 0.0


class PageParser:
    """
    Extracts structured, useful content from raw HTML/text.

    Filters out:
    - Navigation menus, sidebars, footers
    - Ads and tracking scripts
    - Boilerplate text

    Preserves:
    - Main article/content text
    - Headings and structure
    - Code blocks
    - Links with context
    """

    # Common noise patterns to remove
    NOISE_PATTERNS = [
        r"<nav[^>]*>.*?</nav>",
        r"<footer[^>]*>.*?</footer>",
        r"<header[^>]*>.*?</header>",
        r"<aside[^>]*>.*?</aside>",
        r"<script[^>]*>.*?</script>",
        r"<style[^>]*>.*?</style>",
        r"<!--.*?-->",
        r"<div[^>]*class=['\"](?:sidebar|widget|ad|banner|cookie)[^'\"]*['\"][^>]*>.*?</div>",
    ]

    def parse_html(self, html: str, url: str = "") -> ParsedContent:
        """
        Parse HTML content into structured data.

        Args:
            html: Raw HTML string
            url: Source URL (for context)

        Returns:
            ParsedContent with extracted information
        """
        from bs4 import BeautifulSoup

        # Clean HTML first
        cleaned = self._clean_html(html)
        soup = BeautifulSoup(cleaned, "html.parser")

        # Extract components
        title = self._extract_title(soup)
        headings = self._extract_headings(soup)
        paragraphs = self._extract_paragraphs(soup)
        links = self._extract_links(soup, url)
        code_blocks = self._extract_code_blocks(soup)
        metadata = self._extract_metadata(soup)

        main_text = "\n\n".join(paragraphs)
        word_count = len(main_text.split())

        return ParsedContent(
            title=title,
            main_text=main_text,
            headings=headings,
            paragraphs=paragraphs,
            links=links,
            code_blocks=code_blocks,
            metadata=metadata,
            word_count=word_count,
        )

    def parse_text(self, text: str, url: str = "") -> ParsedContent:
        """
        Parse plain text content (e.g., from inner_text).

        Args:
            text: Plain text string
            url: Source URL

        Returns:
            ParsedContent with extracted information
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        paragraphs = []
        headings = []

        for line in lines:
            # Detect headings by capitalization or length
            if len(line) < 80 and line.isupper():
                headings.append({"level": "h2", "text": line})
            elif line.startswith(("#", "##", "###")):
                level = len(line.split(" ")[0])
                text = line.lstrip("#").strip()
                headings.append({"level": f"h{level}", "text": text})
            else:
                paragraphs.append(line)

        main_text = "\n\n".join(paragraphs)
        word_count = len(main_text.split())

        return ParsedContent(
            title=headings[0]["text"] if headings else (lines[0] if lines else ""),
            main_text=main_text,
            headings=headings,
            paragraphs=paragraphs,
            word_count=word_count,
        )

    def rank_relevance(
        self,
        content: ParsedContent,
        query: str,
    ) -> float:
        """
        Score how relevant parsed content is to a query.

        Args:
            content: Parsed page content
            query: Search query

        Returns:
            Relevance score between 0 and 1
        """
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        text = content.main_text.lower()
        title = content.title.lower()

        # Word overlap in text
        text_words = set(text.split())
        text_overlap = len(query_words & text_words) / len(query_words)

        # Word overlap in title (weighted higher)
        title_words = set(title.split())
        title_overlap = len(query_words & title_words) / len(query_words)

        # Length penalty (very short or very long gets penalized)
        wc = content.word_count
        length_factor = 1.0
        if wc < 50:
            length_factor = 0.5
        elif wc > 10000:
            length_factor = 0.8

        score = (0.4 * title_overlap + 0.5 * text_overlap + 0.1) * length_factor
        content.relevance_score = min(1.0, score)
        return content.relevance_score

    def summarize(self, content: ParsedContent, max_sentences: int = 3) -> str:
        """
        Extract a simple summary (first N sentences of main text).

        Args:
            content: Parsed content
            max_sentences: Maximum sentences in summary

        Returns:
            Summary string
        """
        sentences = re.split(r"[.!?]+", content.main_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        summary = ". ".join(sentences[:max_sentences])
        if summary and not summary.endswith("."):
            summary += "."
        return summary

    def _clean_html(self, html: str) -> str:
        """Remove noise patterns from HTML."""
        cleaned = html
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned

    def _extract_title(self, soup) -> str:
        """Extract page title."""
        # Try <title> tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        # Try <h1>
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        # Try og:title
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"]

        return ""

    def _extract_headings(self, soup) -> List[Dict[str, str]]:
        """Extract all headings with their levels."""
        headings = []
        for tag in soup.find_all(re.compile(r"^h[1-6]$")):
            headings.append({
                "level": tag.name,
                "text": tag.get_text(strip=True),
            })
        return headings

    def _extract_paragraphs(self, soup) -> List[str]:
        """Extract main content paragraphs."""
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 20:  # Skip very short paragraphs
                paragraphs.append(text)
        return paragraphs

    def _extract_links(self, soup, base_url: str = "") -> List[Dict[str, str]]:
        """Extract links with their text and href."""
        from urllib.parse import urljoin

        links = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = urljoin(base_url, a["href"]) if base_url else a["href"]
            if text and len(text) > 2:
                links.append({"text": text, "href": href})
        return links

    def _extract_code_blocks(self, soup) -> List[str]:
        """Extract code blocks."""
        code_blocks = []
        for code in soup.find_all("code"):
            text = code.get_text()
            if len(text) > 10:
                code_blocks.append(text)
        for pre in soup.find_all("pre"):
            text = pre.get_text()
            if len(text) > 10 and text not in code_blocks:
                code_blocks.append(text)
        return code_blocks

    def _extract_metadata(self, soup) -> Dict[str, str]:
        """Extract meta tags."""
        meta = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property", "")
            content = tag.get("content", "")
            if name and content:
                meta[name] = content
        return meta
