"""
NICTO AI - Regex Builder Tool
Generate, test, and explain regular expressions.
"""

import re
import logging
from typing import Dict, List, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class RegexBuilderTool(Tool):
    """
    Regular expression builder and tester.

    Features:
    - Generate regex from description
    - Test regex against input
    - Explain regex patterns
    - Common patterns library
    - Match extraction
    """

    name = "regex_builder"
    description = "Build, test, and explain regular expressions. Generate patterns from descriptions and validate against input."
    parameters = [
        ToolParameter(name="pattern", type="string", description="Regex pattern or description", required=True),
        ToolParameter(name="test_string", type="string", description="String to test the pattern against", required=False),
        ToolParameter(name="operation", type="string", description="Operation to perform", required=False, default="test", enum=[
            "test", "match", "findall", "explain", "generate",
        ]),
        ToolParameter(name="flags", type="string", description="Regex flags (e.g., 'i' for case-insensitive)", required=False),
    ]
    tags = ["regex", "pattern", "text", "validation"]
    timeout_seconds = 10.0

    # Common regex patterns library
    COMMON_PATTERNS = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone_us": r'(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        "phone_intl": r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        "url": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "ipv6": r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}',
        "date_iso": r'\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])',
        "date_us": r'(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}',
        "time_24h": r'(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?',
        "time_12h": r'(?:1[0-2]|0?[1-9]):[0-5]\d\s*(?:AM|PM|am|pm)',
        "zip_code_us": r'\b\d{5}(?:-\d{4})?\b',
        "credit_card": r'\b(?:\d[ -]*?){13,16}\b',
        "hex_color": r'(?:#)?[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?',
        "html_tag": r'<([a-z][a-z0-9]*)\b[^>]*>(.*?)</\1>',
        "markdown_link": r'\[([^\]]+)\]\(([^)]+)\)',
        "username": r'[a-zA-Z0-9_]{3,20}',
        "password_strong": r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
        "uuid": r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        "ssn": r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        "isbn": r'(?:ISBN[- ]?)?(?:97[89][ -]?)?\d{1,5}[ -]?\d{1,7}[ -]?\d{1,7}[ -]?\d',
    }

    # Pattern building blocks
    BUILDING_BLOCKS = {
        "digit": r'\d',
        "letter": r'[a-zA-Z]',
        "word": r'\w',
        "space": r'\s',
        "any": r'.',
        "start": r'^',
        "end": r'$',
        "one_or_more": '+',
        "zero_or_more": '*',
        "optional": '?',
        "n_times": '{n}',
        "range": '{m,n}',
        "not": '^',
        "or": '|',
        "group": '(...)',
        "non_capture": '(?:...)',
        "positive_lookahead": '(?=...)',
        "negative_lookahead": '(?!...)',
    }

    def _execute(self, pattern: str, test_string: str = None, operation: str = "test",
                 flags: str = None) -> ToolResult:

        # Check if pattern is a common pattern name
        if pattern in self.COMMON_PATTERNS:
            actual_pattern = self.COMMON_PATTERNS[pattern]
            pattern_name = pattern
        else:
            actual_pattern = pattern
            pattern_name = None

        # Parse flags
        re_flags = self._parse_flags(flags)

        if operation == "generate":
            return self._generate_pattern(pattern)

        if operation == "explain":
            return self._explain_pattern(actual_pattern)

        if not test_string:
            return ToolResult(
                success=True,
                output={
                    "pattern": actual_pattern,
                    "pattern_name": pattern_name,
                    "note": "Pattern compiled successfully. Provide test_string to test.",
                    "common_patterns": list(self.COMMON_PATTERNS.keys()) if not pattern_name else None,
                },
            )

        try:
            compiled = re.compile(actual_pattern, re_flags)

            if operation == "test":
                matches = compiled.findall(test_string)
                return ToolResult(
                    success=True,
                    output={
                        "pattern": actual_pattern,
                        "test_string": test_string,
                        "is_valid": True,
                        "has_matches": bool(matches),
                        "match_count": len(matches),
                        "matches": matches[:20],
                    },
                )
            elif operation == "match":
                match = compiled.search(test_string)
                if match:
                    return ToolResult(
                        success=True,
                        output={
                            "pattern": actual_pattern,
                            "test_string": test_string,
                            "matched": True,
                            "full_match": match.group(),
                            "groups": match.groups(),
                            "start": match.start(),
                            "end": match.end(),
                            "span": match.span(),
                        },
                    )
                else:
                    return ToolResult(
                        success=True,
                        output={
                            "pattern": actual_pattern,
                            "test_string": test_string,
                            "matched": False,
                        },
                    )
            elif operation == "findall":
                matches = compiled.findall(test_string)
                return ToolResult(
                    success=True,
                    output={
                        "pattern": actual_pattern,
                        "test_string": test_string,
                        "matches": matches,
                        "count": len(matches),
                    },
                )

        except re.error as e:
            return ToolResult(
                success=False,
                error=f"Invalid regex: {str(e)}",
                output={"pattern": actual_pattern, "is_valid": False},
            )

    def _generate_pattern(self, description: str) -> ToolResult:
        """Generate regex pattern from description"""
        desc_lower = description.lower()

        # Check common patterns
        for name, pattern in self.COMMON_PATTERNS.items():
            if name in desc_lower:
                return ToolResult(
                    success=True,
                    output={
                        "description": description,
                        "generated_pattern": pattern,
                        "pattern_name": name,
                        "common_patterns_used": True,
                    },
                )

        # Simple pattern generation based on keywords
        if "email" in desc_lower:
            pattern = self.COMMON_PATTERNS["email"]
        elif "phone" in desc_lower:
            pattern = self.COMMON_PATTERNS["phone_us"]
        elif "url" in desc_lower or "link" in desc_lower:
            pattern = self.COMMON_PATTERNS["url"]
        elif "date" in desc_lower:
            pattern = self.COMMON_PATTERNS["date_iso"]
        elif "ip" in desc_lower:
            pattern = self.COMMON_PATTERNS["ip_address"]
        elif "number" in desc_lower:
            pattern = r'\d+'
        elif "word" in desc_lower:
            pattern = r'\b\w+\b'
        else:
            return ToolResult(
                success=True,
                output={
                    "description": description,
                    "generated_pattern": None,
                    "note": "Could not auto-generate. Use building blocks or common patterns.",
                    "common_patterns": list(self.COMMON_PATTERNS.keys()),
                    "building_blocks": self.BUILDING_BLOCKS,
                },
            )

        return ToolResult(
            success=True,
            output={
                "description": description,
                "generated_pattern": pattern,
            },
        )

    def _explain_pattern(self, pattern: str) -> ToolResult:
        """Explain what a regex pattern does"""
        explanations = []

        # Simple token-based explanation
        tokens = [
            (r'\\d', "any digit (0-9)"),
            (r'\\w', "any word character (letter, digit, underscore)"),
            (r'\\s', "any whitespace character"),
            (r'\\b', "word boundary"),
            (r'\.', "literal dot"),
            (r'\+', "one or more of preceding"),
            (r'\*', "zero or more of preceding"),
            (r'\?', "zero or one of preceding (optional)"),
            (r'\[.*?\]', "character class"),
            (r'\(.*?\)', "capturing group"),
            (r'\(?:.*?\)', "non-capturing group"),
            (r'\{.*?\}', "quantifier"),
            (r'\^', "start of string/line"),
            (r'\$', "end of string/line"),
            (r'\|', "OR (alternation)"),
        ]

        for token_pattern, explanation in tokens:
            if re.search(token_pattern, pattern):
                explanations.append(explanation)

        return ToolResult(
            success=True,
            output={
                "pattern": pattern,
                "explanations": explanations if explanations else ["No specific tokens identified"],
                "full_pattern": pattern,
            },
        )

    def _parse_flags(self, flags: str) -> int:
        """Parse regex flags"""
        flag_map = {
            'i': re.IGNORECASE,
            'm': re.MULTILINE,
            's': re.DOTALL,
            'x': re.VERBOSE,
        }
        result = 0
        if flags:
            for char in flags.lower():
                if char in flag_map:
                    result |= flag_map[char]
        return result
