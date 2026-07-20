"""
Experience Buffer for CSET
==========================
Stores and retrieves reasoning experiences for cumulative learning.
Working → Episodic → Semantic memory consolidation.
"""
import sys, json, time, math
from pathlib import Path
from typing import List, Dict, Optional
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


class Experience:
    """A single reasoning experience."""

    def __init__(self, problem: str, solution: str, reasoning_steps: List[str],
                 correct: bool, domain: str = "general", difficulty: float = 0.5,
                 reward_scores: Dict = None):
        self.problem = problem
        self.solution = solution
        self.reasoning_steps = reasoning_steps
        self.correct = correct
        self.domain = domain
        self.difficulty = difficulty
        self.reward_scores = reward_scores or {}
        self.timestamp = time.time()
        self.access_count = 0
        self.usefulness = 0.0

    def to_dict(self) -> Dict:
        return {
            "problem": self.problem,
            "solution": self.solution,
            "reasoning_steps": self.reasoning_steps,
            "correct": self.correct,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "reward_scores": self.reward_scores,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "usefulness": self.usefulness,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Experience":
        exp = cls(
            problem=d["problem"],
            solution=d["solution"],
            reasoning_steps=d["reasoning_steps"],
            correct=d["correct"],
            domain=d.get("domain", "general"),
            difficulty=d.get("difficulty", 0.5),
            reward_scores=d.get("reward_scores", {}),
        )
        exp.timestamp = d.get("timestamp", time.time())
        exp.access_count = d.get("access_count", 0)
        exp.usefulness = d.get("usefulness", 0.0)
        return exp


class WorkingMemory:
    """Recent experiences (last N). Always accessible."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, experience: Experience):
        self.buffer.append(experience)

    def get_recent(self, k: int = 10) -> List[Experience]:
        return list(self.buffer)[-k:]

    def __len__(self):
        return len(self.buffer)


class EpisodicMemory:
    """Medium-term experiences with similarity search."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.experiences: List[Experience] = []

    def store(self, experience: Experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.capacity:
            self.experiences.sort(key=lambda e: e.usefulness, reverse=True)
            self.experiences = self.experiences[:self.capacity]

    def search(self, query_domain: str = None, query_difficulty: float = None,
               k: int = 5) -> List[Experience]:
        scored = []
        for exp in self.experiences:
            score = 0.0
            if query_domain and exp.domain == query_domain:
                score += 1.0
            if query_difficulty is not None:
                diff = abs(exp.difficulty - query_difficulty)
                score += max(0, 1.0 - diff)
            if exp.correct:
                score += 0.5
            score += exp.usefulness * 0.3
            scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [exp for _, exp in scored[:k]]
        for exp in results:
            exp.access_count += 1
        return results

    def __len__(self):
        return len(self.experiences)


class SemanticMemory:
    """Long-term abstracted patterns."""

    def __init__(self):
        self.patterns: Dict[str, List[str]] = {}

    def store_pattern(self, domain: str, approach: str):
        if domain not in self.patterns:
            self.patterns[domain] = []
        if approach not in self.patterns[domain]:
            self.patterns[domain].append(approach)

    def get_patterns(self, domain: str, k: int = 3) -> List[str]:
        return self.patterns.get(domain, [])[:k]

    def consolidate(self, episodic: EpisodicMemory):
        for exp in episodic.experiences:
            if exp.correct and exp.usefulness > 0.5:
                approach = " -> ".join(exp.reasoning_steps[:3])
                self.store_pattern(exp.domain, approach)

    def __len__(self):
        return sum(len(v) for v in self.patterns.values())


class ExperienceBuffer:
    """
    Full experience buffer with 3-tier memory.
    Integrates with NICTO's HierarchicalMemory.
    """

    def __init__(self, working_capacity: int = 100, episodic_capacity: int = 1000):
        self.working = WorkingMemory(working_capacity)
        self.episodic = EpisodicMemory(episodic_capacity)
        self.semantic = SemanticMemory()
        self._consolidation_threshold = 10
        self._unconsolidated = 0

    def store(self, experience: Experience):
        self.working.store(experience)
        self.episodic.store(experience)
        self._unconsolidated += 1

        if self._unconsolidated >= self._consolidation_threshold:
            self.semantic.consolidate(self.episodic)
            self._unconsolidated = 0

    def retrieve(self, domain: str = None, difficulty: float = None,
                 k: int = 10) -> Dict:
        recent = self.working.get_recent(k // 2)
        episodic = self.episodic.search(domain, difficulty, k // 2)
        patterns = self.semantic.get_patterns(domain or "general")

        all_experiences = recent + episodic
        for exp in all_experiences:
            exp.usefulness += 0.01

        return {
            "recent": recent,
            "episodic": episodic,
            "patterns": patterns,
            "total": len(recent) + len(episodic),
        }

    def get_hints(self, problem_domain: str, difficulty: float) -> str:
        retrieved = self.retrieve(domain=problem_domain, difficulty=difficulty, k=5)
        hints = []
        for exp in retrieved["recent"] + retrieved["episodic"]:
            if exp.correct and exp.difficulty >= difficulty * 0.7:
                hint = f"Similar solved problem: {' -> '.join(exp.reasoning_steps[:2])}"
                hints.append(hint)
        for pattern in retrieved["patterns"]:
            hints.append(f"Known approach: {pattern}")

        return "\n".join(hints[:5])

    def stats(self) -> Dict:
        return {
            "working": len(self.working),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "total": len(self.working) + len(self.episodic) + len(self.semantic),
        }

    def save(self, path: str):
        data = {
            "episodic": [e.to_dict() for e in self.episodic.experiences],
            "semantic": self.semantic.patterns,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("episodic", []):
            self.episodic.store(Experience.from_dict(d))
        self.semantic.patterns = data.get("semantic", {})
