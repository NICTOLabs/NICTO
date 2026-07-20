"""
Disciplined Chain-of-Thought (D-CoT) Tags for CSET
====================================================
Control tags that guide reasoning depth during training.
Inspired by D-CoT (arXiv:2602.21786).
Tags act as training wheels — the model internalizes them over time.
"""
import sys, random
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Control tags: each guides a different reasoning mode
CONTROL_TAGS = {
    "TEMP_LOW": {
        "description": "Careful fact-checking and verification",
        "use_when": "Verifying claims, checking facts, ensuring accuracy",
        "examples": [
            "Let me carefully verify each part of this claim.",
            "Fact-checking step by step...",
            "Is this actually true? Let me check the evidence.",
        ],
    },
    "TEMP_HIGH": {
        "description": "Multi-perspective exploration",
        "use_when": "Brainstorming, creative problem-solving, exploring alternatives",
        "examples": [
            "Let me explore multiple perspectives on this.",
            "What are the different ways to approach this?",
            "Considering various angles...",
        ],
    },
    "REASON": {
        "description": "Logical deduction and inference",
        "use_when": "Drawing conclusions, logical reasoning, causal analysis",
        "examples": [
            "Following the logical chain...",
            "Therefore, we can conclude that...",
            "The logical structure implies...",
        ],
    },
    "PLAN": {
        "description": "Strategic planning and decomposition",
        "use_when": "Breaking down complex problems, planning multi-step solutions",
        "examples": [
            "Let me break this down into steps.",
            "Step 1: Identify the key components.",
            "Planning the approach...",
        ],
    },
    "CRITIQUE": {
        "description": "Self-criticism and error detection",
        "use_when": "Reviewing work, finding flaws, improving quality",
        "examples": [
            "Let me check for potential errors in my reasoning.",
            "Is there a flaw in this argument?",
            "Self-critique: what could be wrong?",
        ],
    },
    "SYNTHESIZE": {
        "description": "Combining multiple pieces of information",
        "use_when": "Integrating knowledge, cross-domain connections",
        "examples": [
            "Combining these insights...",
            "Synthesizing information from multiple sources.",
            "The connection between these ideas is...",
        ],
    },
}

DOMAIN_TAG_PREFERENCES = {
    "logic": ["REASON", "CRITIQUE", "TEMP_LOW"],
    "math": ["PLAN", "REASON", "TEMP_LOW"],
    "code": ["PLAN", "CRITIQUE", "TEMP_LOW"],
    "science": ["TEMP_LOW", "SYNTHESIZE", "REASON"],
    "research": ["SYNTHESIZE", "TEMP_HIGH", "CRITIQUE"],
    "ethics": ["TEMP_HIGH", "CRITIQUE", "SYNTHESIZE"],
    "creative": ["TEMP_HIGH", "SYNTHESIZE", "PLAN"],
    "analysis": ["PLAN", "REASON", "CRITIQUE"],
}

DIFFICULTY_TAG_COUNT = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
}


class DCoTPrompter:
    """Generates D-CoT tagged prompts for training."""

    def __init__(self):
        self.tags_used = {tag: 0 for tag in CONTROL_TAGS}

    def add_tags(self, problem_text: str, domain: str, difficulty: float,
                 num_tags: int = None) -> str:
        level = max(1, min(5, int(difficulty * 5) + 1))
        if num_tags is None:
            num_tags = DIFFICULTY_TAG_COUNT.get(level, 1)

        preferred = DOMAIN_TAG_PREFERENCES.get(domain, ["REASON"])
        available = [t for t in CONTROL_TAGS if t not in preferred[:num_tags]]
        selected_tags = preferred[:num_tags]
        while len(selected_tags) < num_tags and available:
            tag = random.choice(available)
            selected_tags.append(tag)
            available.remove(tag)

        prompt_parts = []
        for tag in selected_tags:
            tag_info = CONTROL_TAGS[tag]
            example = random.choice(tag_info["examples"])
            prompt_parts.append(f"<{tag}>{example}</{tag}>")

        prompt_parts.append(f"\nProblem: {problem_text}")

        return "\n".join(prompt_parts)

    def strip_tags(self, text: str) -> str:
        import re
        return re.sub(r'<[^>]+>[^<]*</[^>]+>', '', text).strip()

    def extract_tags(self, text: str) -> List[str]:
        import re
        return re.findall(r'<(\w+)>', text)

    def get_tag_stats(self) -> Dict:
        return dict(self.tags_used)


class DifficultyScaler:
    """Scales training difficulty over time."""

    def __init__(self, initial: float = 0.2, max_difficulty: float = 1.0,
                 warmup_steps: int = 1000, ramp_steps: int = 10000):
        self.initial = initial
        self.max = max_difficulty
        self.warmup = warmup_steps
        self.ramp = ramp_steps
        self.step = 0

    def get_difficulty(self) -> float:
        if self.step < self.warmup:
            progress = self.step / self.warmup
            return self.initial * progress
        else:
            elapsed = self.step - self.warmup
            progress = min(1.0, elapsed / self.ramp)
            return self.initial + (self.max - self.initial) * progress

    def advance(self):
        self.step += 1

    def status(self) -> str:
        return f"step={self.step} difficulty={self.get_difficulty():.3f}"
