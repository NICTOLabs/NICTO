"""
NICTO AI - Content Writer Tool
Generate blog posts, emails, articles, social media content, and more.
"""

import logging
from typing import Dict, Optional
from .base import Tool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class ContentWriterTool(Tool):
    """
    Content generation engine.

    Creates high-quality content in various formats:
    - Blog posts and articles
    - Email sequences
    - Social media posts
    - Product descriptions
    - Ad copy
    - Press releases
    - Technical documentation
    """

    name = "content_writer"
    description = "Generate high-quality content: blog posts, emails, articles, social media, product descriptions, ad copy."
    parameters = [
        ToolParameter(name="topic", type="string", description="Topic or subject to write about", required=True),
        ToolParameter(name="content_type", type="string", description="Type of content to generate", required=True, enum=[
            "blog_post", "article", "email", "social_media", "product_description",
            "ad_copy", "press_release", "technical_doc", "newsletter", "speech",
            "story", "poem", "resume", "cover_letter", "business_plan",
        ]),
        ToolParameter(name="tone", type="string", description="Writing tone", required=False, default="professional", enum=[
            "professional", "casual", "formal", "friendly", "persuasive",
            "humorous", "technical", "academic", "creative", "inspirational",
        ]),
        ToolParameter(name="length", type="string", description="Content length", required=False, default="medium", enum=[
            "short", "medium", "long", "detailed",
        ]),
        ToolParameter(name="audience", type="string", description="Target audience", required=False),
        ToolParameter(name="keywords", type="string", description="Keywords to include (comma-separated)", required=False),
        ToolParameter(name="extra_instructions", type="string", description="Additional instructions", required=False),
    ]
    tags = ["content", "writing", "copywriting", "blog", "email"]
    timeout_seconds = 30.0

    # Content templates with structure guides
    TEMPLATES = {
        "blog_post": {
            "structure": "Hook → Introduction → Problem → Solution → Benefits → CTA → Conclusion",
            "elements": ["attention-grabbing title", "clear thesis", "subheadings", "bullet points", "conclusion with CTA"],
        },
        "article": {
            "structure": "Headline → Lead → Body (inverted pyramid) → Conclusion",
            "elements": ["factual tone", "quotes", "data points", "smooth transitions"],
        },
        "email": {
            "structure": "Subject line → Greeting → Hook → Body → CTA → Sign-off",
            "elements": ["compelling subject", "personalization", "clear CTA", "scannable format"],
        },
        "social_media": {
            "structure": "Hook → Value → Engagement → CTA",
            "elements": ["emoji", "hashtags", "call to action", "brevity"],
        },
        "product_description": {
            "structure": "Headline → Features → Benefits → Social proof → CTA",
            "elements": ["sensory language", "unique selling points", "urgency"],
        },
        "ad_copy": {
            "structure": "Attention → Interest → Desire → Action",
            "elements": ["power words", "numbers", "urgency", "social proof"],
        },
        "technical_doc": {
            "structure": "Overview → Prerequisites → Steps → Examples → Troubleshooting",
            "elements": ["code blocks", "clear steps", "examples", "warnings"],
        },
        "resume": {
            "structure": "Contact → Summary → Experience → Skills → Education",
            "elements": ["action verbs", "quantified achievements", "ATS-friendly"],
        },
        "cover_letter": {
            "structure": "Opening → Why you → Why them → CTA",
            "elements": ["specific examples", "company research", "enthusiasm"],
        },
        "business_plan": {
            "structure": "Executive Summary → Market → Product → Marketing → Finance → Team",
            "elements": ["market analysis", "revenue model", "competitive advantage"],
        },
    }

    def _execute(self, topic: str, content_type: str, tone: str = "professional",
                 length: str = "medium", audience: str = None, keywords: str = None,
                 extra_instructions: str = None) -> ToolResult:

        template = self.TEMPLATES.get(content_type, {})
        structure = template.get("structure", "")
        elements = template.get("elements", [])

        length_guide = {
            "short": "2-3 paragraphs, ~150-200 words",
            "medium": "4-6 paragraphs, ~400-600 words",
            "long": "8-12 paragraphs, ~800-1200 words",
            "detailed": "15+ paragraphs, ~1500-2500 words with sections",
        }

        output = {
            "topic": topic,
            "content_type": content_type,
            "tone": tone,
            "length": length_guide.get(length, length),
            "structure": structure,
            "elements_to_include": elements,
            "audience": audience or "general",
            "keywords": keywords.split(",") if keywords else [],
            "extra_instructions": extra_instructions or "",
            "note": "ContentWriter generates structure and guidelines. For actual text generation, connect to NICTO model.",
        }

        # Generate a content outline
        outline = self._generate_outline(topic, content_type, tone, length, audience)
        output["outline"] = outline

        return ToolResult(success=True, output=output)

    def _generate_outline(self, topic: str, content_type: str, tone: str,
                          length: str, audience: str) -> Dict:
        """Generate a content outline based on parameters"""
        outline = {
            "title_suggestions": [
                f"The Ultimate Guide to {topic.title()}",
                f"How to Master {topic.title()}: A Complete Guide",
                f"{topic.title()}: Everything You Need to Know",
            ],
            "sections": [],
        }

        if content_type == "blog_post":
            outline["sections"] = [
                {"heading": f"What is {topic}?", "purpose": "Introduction and definition"},
                {"heading": f"Why {topic} Matters", "purpose": "Importance and relevance"},
                {"heading": f"Key Benefits of {topic}", "purpose": "Value proposition"},
                {"heading": f"How to Get Started with {topic}", "purpose": "Actionable steps"},
                {"heading": f"Common Mistakes to Avoid", "purpose": "Pitfalls and warnings"},
                {"heading": f"Conclusion and Next Steps", "purpose": "Summary and CTA"},
            ]
        elif content_type == "email":
            outline["sections"] = [
                {"heading": "Subject Line", "purpose": "Compelling open"},
                {"heading": "Opening Hook", "purpose": "Personal connection"},
                {"heading": "Core Message", "purpose": "Main value/content"},
                {"heading": "Call to Action", "purpose": "Next steps"},
            ]
        elif content_type == "product_description":
            outline["sections"] = [
                {"heading": "Headline", "purpose": "Attention grabber"},
                {"heading": "Key Features", "purpose": "What it does"},
                {"heading": "Benefits", "purpose": "Why it matters"},
                {"heading": "Social Proof", "purpose": "Trust building"},
                {"heading": "CTA", "purpose": "Purchase trigger"},
            ]
        else:
            outline["sections"] = [
                {"heading": "Introduction", "purpose": "Set context"},
                {"heading": "Main Content", "purpose": "Core message"},
                {"heading": "Conclusion", "purpose": "Summary and CTA"},
            ]

        return outline
