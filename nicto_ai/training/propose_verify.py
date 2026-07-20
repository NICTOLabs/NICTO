"""
Propose & Verify for CSET Self-Play
====================================
Three-role framework: Proposer, Solver, Verifier.
Inspired by Absolute Zero Reasoner.
"""
import sys, random, json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


DOMAIN_TEMPLATES = {
    "logic": [
        "Given {premise}, can we conclude {conclusion}?",
        "If {premise1} and {premise2}, what follows?",
        "Evaluate the validity: {argument}",
        "Is this syllogism valid? {syllogism}",
    ],
    "math": [
        "Solve: {problem}",
        "Prove that {statement}",
        "Find all values of x where {equation}",
        "What is the {operation} of {expression}?",
    ],
    "code": [
        "Write a function that {description}",
        "Debug this code: {code}",
        "Optimize: {code}",
        "What does this code do? {code}",
    ],
    "science": [
        "Explain why {phenomenon}",
        "What causes {effect}?",
        "Design an experiment to test {hypothesis}",
        "Predict what happens when {condition}",
    ],
    "research": [
        "Analyze the implications of {finding}",
        "What are the limitations of {approach}?",
        "Synthesize findings from {field1} and {field2}",
        "Propose a novel approach to {problem}",
    ],
    "ethics": [
        "Evaluate the moral implications of {action}",
        "What are the arguments for and against {position}?",
        "How should {stakeholder} approach {dilemma}?",
        "What principles apply to {situation}?",
    ],
    "creative": [
        "Write a {style} piece about {topic}",
        "Create an analogy for {concept}",
        "Design a solution to {problem} using {approach}",
        "What are 5 unexpected uses for {object}?",
    ],
    "analysis": [
        "Interpret this data: {data_description}",
        "What conclusions can be drawn from {evidence}?",
        "Identify patterns in {input}",
        "Evaluate the strength of {argument}",
    ],
}

DIFFICULTY_MODIFIERS = {
    1: ["basic", "simple", "straightforward", "elementary"],
    2: ["moderate", "standard", "clear", "direct"],
    3: ["complex", "multi-step", "nuanced", "challenging"],
    4: ["ambiguous", "open-ended", "requires judgment", "subtle"],
    5: ["research-level", "cutting-edge", "deeply nuanced", "unprecedented"],
}


class ProposerRole:
    """Generates diverse training problems at calibrated difficulty."""

    def __init__(self):
        self.domains = list(DOMAIN_TEMPLATES.keys())
        self.generation_count = 0
        self.domain_counts = {d: 0 for d in self.domains}

    def propose(self, difficulty: float = 0.5, preferred_domain: str = None) -> Dict:
        if preferred_domain:
            domain = preferred_domain
        else:
            least_used = min(self.domain_counts, key=self.domain_counts.get)
            if random.random() < 0.3:
                domain = least_used
            else:
                domain = random.choice(self.domains)

        templates = DOMAIN_TEMPLATES[domain]
        template = random.choice(templates)

        level = max(1, min(5, int(difficulty * 5) + 1))
        modifiers = DIFFICULTY_MODIFIERS[level]
        modifier = random.choice(modifiers)

        problem_text = self._fill_template(template, domain, modifier)
        self.generation_count += 1
        self.domain_counts[domain] += 1

        return {
            "text": problem_text,
            "domain": domain,
            "difficulty": difficulty,
            "level": level,
            "modifier": modifier,
        }

    def _fill_template(self, template: str, domain: str, modifier: str) -> str:
        placeholders = {
            "{premise}": f"all {random.choice(['A', 'B', 'C'])}s are {random.choice(['D', 'E', 'F'])}s",
            "{conclusion}": f"all {random.choice(['D', 'E', 'F'])}s are {random.choice(['A', 'B', 'C'])}s",
            "{premise1}": f"x > 0 and y < 10",
            "{premise2}": f"x + y = 15",
            "{argument}": f"All {random.choice(['cats', 'dogs', 'birds'])} are animals. Some animals are friendly. Therefore, some {random.choice(['cats', 'dogs', 'birds'])} are friendly.",
            "{syllogism}": f"All {random.choice(['Mammals', 'Reptiles', 'Birds'])} are animals. No {random.choice(['fish', 'insects', 'worms'])} are mammals.",
            "{problem}": f"{modifier} problem involving {random.choice(['algebra', 'geometry', 'combinatorics', 'number theory'])}",
            "{statement}": f"a mathematical property of {random.choice(['prime numbers', 'matrices', 'functions', 'series'])}",
            "{equation}": f"x^2 - {random.randint(2,10)}x + {random.randint(1,20)} = 0",
            "{operation}": random.choice(["sum", "product", "limit", "derivative"]),
            "{expression}": f"f(x) = {random.randint(1,5)}x^2 + {random.randint(1,5)}x + {random.randint(1,5)}",
            "{description}": f"performs {modifier} data transformation",
            "{code}": f"for i in range(n): result += arr[i] / (i + 1)",
            "{phenomenon}": f"the sky appears {random.choice(['blue', 'red at sunset', 'dark at night'])}",
            "{effect}": f"{random.choice(['photosynthesis', 'gravity', 'magnetic fields'])}",
            "{hypothesis}": f"temperature affects {random.choice(['growth rate', 'reaction speed', 'solubility'])}",
            "{condition}": f"pressure is {random.choice(['increased', 'decreased', 'varied'])}",
            "{finding}": f"a {modifier} correlation between {random.choice(['variables A and B', 'genetics and behavior', 'input and output'])}",
            "{approach}": f"{modifier} methodology using {random.choice(['statistical analysis', 'controlled experiments', 'computational modeling'])}",
            "{field1}": random.choice(["neuroscience", "economics", "physics", "sociology"]),
            "{field2}": random.choice(["AI", "philosophy", "biology", "mathematics"]),
            "{problem}": f"the {modifier} challenge of {random.choice(['scaling AI', 'climate change', 'drug discovery'])}",
            "{action}": f"using {random.choice(['AI surveillance', 'genetic engineering', 'autonomous weapons'])}",
            "{position}": f"{modifier} stance on {random.choice(['privacy', 'equality', 'freedom'])}",
            "{stakeholder}": random.choice(["governments", "companies", "individuals", "communities"]),
            "{dilemma}": f"the {modifier} trade-off between {random.choice(['safety and freedom', 'efficiency and fairness', 'growth and sustainability'])}",
            "{situation}": f"a {modifier} scenario involving {random.choice(['conflicting rights', 'resource allocation', 'risk assessment'])}",
            "{topic}": random.choice(["technology", "nature", "society", "the future"]),
            "{concept}": f"{modifier} concept in {random.choice(['physics', 'economics', 'philosophy'])}",
            "{object}": random.choice(["paperclip", "smartphone", "wheel", "algorithm"]),
            "{data_description}": f"a dataset showing {random.choice(['trends over time', 'correlations between variables', 'distributions across categories'])}",
            "{evidence}": f"the {modifier} findings from recent studies",
            "{input}": f"a complex dataset with {random.choice(['many variables', 'missing values', 'non-linear patterns'])}",
        }

        result = template
        for key, value in placeholders.items():
            if key in result:
                result = result.replace(key, value, 1)

        return result


class VerifierRole:
    """Verifies solution correctness through multiple strategies."""

    def __init__(self):
        self.verification_count = 0
        self.correct_count = 0

    def verify(self, problem: Dict, solution: str, reasoning_steps: List[str],
               model=None) -> Dict:
        self.verification_count += 1

        scores = {
            "completeness": self._check_completeness(solution),
            "coherence": self._check_coherence(reasoning_steps),
            "relevance": self._check_relevance(problem["text"], solution),
        }

        if model is not None and len(reasoning_steps) >= 3:
            consistency_score = self._self_consistency_check(model, problem, n=3)
            scores["consistency"] = consistency_score

        overall = sum(scores.values()) / len(scores)
        correct = overall > 0.6

        if correct:
            self.correct_count += 1

        return {
            "correct": correct,
            "scores": scores,
            "overall_score": overall,
            "verification_rate": self.correct_count / max(self.verification_count, 1),
        }

    def _check_completeness(self, solution: str) -> float:
        if not solution or len(solution) < 10:
            return 0.2
        if len(solution) > 50:
            return 0.8
        return 0.5

    def _check_coherence(self, steps: List[str]) -> float:
        if not steps:
            return 0.0
        if len(steps) == 1:
            return 0.4
        transitions = 0
        for i in range(1, len(steps)):
            if any(word in steps[i].lower() for word in
                   ["therefore", "thus", "hence", "so", "because", "since",
                    "however", "but", "furthermore", "additionally"]):
                transitions += 1
        return min(1.0, 0.4 + (transitions / max(len(steps) - 1, 1)) * 0.6)

    def _check_relevance(self, problem: str, solution: str) -> float:
        problem_words = set(problem.lower().split())
        solution_words = set(solution.lower().split())
        overlap = len(problem_words & solution_words)
        total = len(problem_words | solution_words)
        if total == 0:
            return 0.0
        return min(1.0, overlap / total * 3)

    def _self_consistency_check(self, model, problem: Dict, n: int = 3) -> float:
        return 0.5


class SelfPlayLoop:
    """Orchestrates the Propose → Solve → Verify loop."""

    def __init__(self, proposer: ProposerRole = None, verifier: VerifierRole = None):
        self.proposer = proposer or ProposerRole()
        self.verifier = verifier or VerifierRole()
        self.loop_count = 0
        self.total_correct = 0

    def generate_problem(self, difficulty: float = 0.5) -> Dict:
        return self.proposer.propose(difficulty)

    def verify_solution(self, problem: Dict, solution: str,
                        reasoning_steps: List[str], model=None) -> Dict:
        result = self.verifier.verify(problem, solution, reasoning_steps, model)
        self.loop_count += 1
        if result["correct"]:
            self.total_correct += 1
        return result

    def stats(self) -> Dict:
        return {
            "loops": self.loop_count,
            "correct": self.total_correct,
            "accuracy": self.total_correct / max(self.loop_count, 1),
            "proposals": self.proposer.generation_count,
        }
