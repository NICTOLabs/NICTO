"""
Reasoning Evolver for CSET
==========================
Mutates and recombines reasoning chains to find smarter approaches.
Inspired by CoT-Evo and PopuLoRA evolutionary training.
"""
import sys, random, copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn


class ReasoningChain:
    """A single reasoning chain with steps and metadata."""

    def __init__(self, steps: List[str] = None, domain: str = "general"):
        self.steps = steps or []
        self.domain = domain
        self.scores = []
        self.correct = False
        self.fitness = 0.0

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"ReasoningChain(domain={self.domain}, steps={len(self.steps)}, fitness={self.fitness:.3f})"


class MutationOperators:
    """Collection of reasoning chain mutation operations."""

    @staticmethod
    def swap_steps(chain: ReasoningChain) -> ReasoningChain:
        if len(chain.steps) < 2:
            return chain
        new = copy.deepcopy(chain)
        i, j = random.sample(range(len(new.steps)), 2)
        new.steps[i], new.steps[j] = new.steps[j], new.steps[i]
        return new

    @staticmethod
    def insert_step(chain: ReasoningChain) -> ReasoningChain:
        new = copy.deepcopy(chain)
        template_steps = [
            "Let me verify this step by checking the logic.",
            "Considering alternative perspectives...",
            "This can be cross-checked with known facts.",
            "Let me break this down further.",
            "Checking for potential errors in reasoning.",
        ]
        step = random.choice(template_steps)
        pos = random.randint(0, len(new.steps))
        new.steps.insert(pos, step)
        return new

    @staticmethod
    def remove_step(chain: ReasoningChain) -> ReasoningChain:
        if len(chain.steps) <= 1:
            return chain
        new = copy.deepcopy(chain)
        idx = random.randint(0, len(new.steps) - 1)
        new.steps.pop(idx)
        return new

    @staticmethod
    def add_verification(chain: ReasoningChain) -> ReasoningChain:
        new = copy.deepcopy(chain)
        verify_step = "Let me verify: does this conclusion follow from the premises?"
        new.steps.append(verify_step)
        return new

    @staticmethod
    def simplify(chain: ReasoningChain) -> ReasoningChain:
        if len(chain.steps) <= 1:
            return chain
        new = copy.deepcopy(chain)
        mid = len(new.steps) // 2
        new.steps = new.steps[:mid] + new.steps[mid+1:]
        return new

    @staticmethod
    def deepen(chain: ReasoningChain) -> ReasoningChain:
        new = copy.deepcopy(chain)
        detail = "Going deeper: what are the implications of this step?"
        idx = random.randint(0, len(new.steps) - 1)
        new.steps.insert(idx + 1, detail)
        return new


class Population:
    """Manages a population of reasoning chains for evolutionary selection."""

    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.chains: List[ReasoningChain] = []

    def add(self, chain: ReasoningChain):
        self.chains.append(chain)
        if len(self.chains) > self.max_size:
            self.chains.sort(key=lambda c: c.fitness, reverse=True)
            self.chains = self.chains[:self.max_size]

    def select_parents(self, n: int = 2) -> List[ReasoningChain]:
        tournament_size = min(3, len(self.chains))
        parents = []
        for _ in range(n):
            candidates = random.sample(self.chains, tournament_size)
            winner = max(candidates, key=lambda c: c.fitness)
            parents.append(winner)
        return parents

    def get_diversity(self) -> float:
        if len(self.chains) < 2:
            return 0.0
        domains = set(c.domain for c in self.chains)
        lengths = [len(c.steps) for c in self.chains]
        domain_div = len(domains) / max(self.max_size, 1)
        len_div = (max(lengths) - min(lengths)) / max(max(lengths), 1) if max(lengths) > 0 else 0
        return (domain_div + len_div) / 2

    def stats(self) -> Dict:
        if not self.chains:
            return {"size": 0, "avg_fitness": 0, "diversity": 0}
        return {
            "size": len(self.chains),
            "avg_fitness": sum(c.fitness for c in self.chains) / len(self.chains),
            "best_fitness": max(c.fitness for c in self.chains),
            "diversity": self.get_diversity(),
        }


class ReasoningEvolver:
    """
    Evolves reasoning chains through mutation, recombination, and selection.
    """

    def __init__(self, pop_size: int = 30, mutation_rate: float = 0.3):
        self.population = Population(max_size=pop_size)
        self.mutation_rate = mutation_rate
        self.operators = MutationOperators()
        self.generation = 0
        self.total_evolved = 0

    def mutate(self, chain: ReasoningChain) -> ReasoningChain:
        ops = [
            self.operators.swap_steps,
            self.operators.insert_step,
            self.operators.remove_step,
            self.operators.add_verification,
            self.operators.simplify,
            self.operators.deepen,
        ]
        op = random.choice(ops)
        return op(chain)

    def recombine(self, parent_a: ReasoningChain, parent_b: ReasoningChain) -> ReasoningChain:
        child = ReasoningChain(domain=parent_a.domain)
        all_steps = list(zip(
            parent_a.steps + [None] * len(parent_b.steps),
            parent_b.steps + [None] * len(parent_a.steps),
        ))

        for step_a, step_b in all_steps[:max(len(parent_a), len(parent_b))]:
            if step_a and step_b:
                child.steps.append(random.choice([step_a, step_b]))
            elif step_a:
                child.steps.append(step_a)
            elif step_b:
                child.steps.append(step_b)

        return child

    def evolve(self, fitness_fn=None) -> ReasoningChain:
        self.generation += 1

        if len(self.population.chains) < 2:
            return ReasoningChain()

        parents = self.population.select_parents(2)

        child = self.recombine(parents[0], parents[1])

        if random.random() < self.mutation_rate:
            child = self.mutate(child)

        if fitness_fn:
            child.fitness = fitness_fn(child)

        self.population.add(child)
        self.total_evolved += 1

        return child

    def update_fitness(self, chain: ReasoningChain, fitness: float):
        chain.fitness = fitness
        self.population.add(chain)

    def get_best(self) -> Optional[ReasoningChain]:
        if not self.population.chains:
            return None
        return max(self.population.chains, key=lambda c: c.fitness)

    def stats(self) -> Dict:
        pop_stats = self.population.stats()
        return {
            "generation": self.generation,
            "total_evolved": self.total_evolved,
            **pop_stats,
        }
