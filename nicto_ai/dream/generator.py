"""
NICTO AI - Synthetic Data Generator
Generates training data from NICTO's own reasoning and experiences

This is how NICTO "dreams" - it generates synthetic training examples
from its own processing, creating new data from existing knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DreamSample:
    """A synthetic training sample generated during dreaming"""
    input_ids: torch.Tensor
    labels: torch.Tensor
    source: str  # "replay", "augmentation", "chain", "contrastive"
    confidence: float = 1.0


class SyntheticDataGenerator:
    """
    Generates synthetic training data from NICTO's experiences.

    Techniques:
    1. Experience Augmentation - Perturb existing experiences to create variations
    2. Reasoning Chain Generation - Create step-by-step reasoning sequences
    3. Contrastive Pairs - Generate positive/negative examples
    4. Knowledge Distillation - Generate soft-label training data from own outputs
    5. Curriculum Sequences - Generate progressively harder examples
    """

    def __init__(self, vocab_size: int = 32000, dim: int = 256):
        self.vocab_size = vocab_size
        self.dim = dim

    def augment_experience(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        n_augments: int = 3,
    ) -> List[DreamSample]:
        """
        Create variations of an existing experience.

        Methods:
        - Token replacement: swap low-confidence tokens
        - Sequence truncation: shorter versions
        - Token reordering: slightly permute positions
        """
        samples = []
        batch_size, seq_len = input_ids.shape

        for _ in range(n_augments):
            aug_input = input_ids.clone()

            # Method 1: Replace low-confidence tokens with random alternatives
            probs = F.softmax(logits, dim=-1)  # [B, seq, vocab]
            max_probs = probs.max(dim=-1).values  # [B, seq]
            low_conf_mask = max_probs < 0.3  # Low confidence positions

            random_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            replace_mask = low_conf_mask & (torch.rand_like(max_probs) < 0.3)
            aug_input[replace_mask] = random_tokens[replace_mask]

            # Method 2: Slight position permutation (swap adjacent tokens)
            if seq_len > 2 and torch.rand(1).item() < 0.5:
                swap_idx = torch.randint(0, seq_len - 1, (1,)).item()
                aug_input[:, swap_idx], aug_input[:, swap_idx + 1] = (
                    aug_input[:, swap_idx + 1].clone(),
                    aug_input[:, swap_idx].clone(),
                )

            # Labels are shifted input
            labels = aug_input.clone()
            samples.append(DreamSample(
                input_ids=aug_input,
                labels=labels,
                source="augmentation",
                confidence=0.7,
            ))

        return samples

    def generate_reasoning_chain(
        self,
        prompt: torch.Tensor,
        model: nn.Module,
        chain_length: int = 5,
        temperature: float = 0.8,
    ) -> DreamSample:
        """
        Generate a reasoning chain (step-by-step thinking) from a prompt.

        The model generates intermediate reasoning steps, creating
        a chain-of-thought training example.
        """
        model.eval()
        generated = [prompt]

        with torch.no_grad():
            current = prompt
            for step in range(chain_length):
                outputs = model(current)
                logits = outputs["logits"][:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current = torch.cat([current, next_token], dim=-1)
                generated.append(next_token)

        model.train()

        full_sequence = torch.cat(generated, dim=-1)
        input_ids = full_sequence[:, :-1]
        labels = full_sequence[:, 1:]

        return DreamSample(
            input_ids=input_ids,
            labels=labels,
            source="chain",
            confidence=0.8,
        )

    def generate_contrastive_pairs(
        self,
        input_ids: torch.Tensor,
        model: nn.Module,
        n_pairs: int = 4,
    ) -> List[DreamSample]:
        """
        Generate contrastive pairs: (good_response, bad_response).

        Creates training signal by showing the model both correct and
        incorrect completions, teaching it to distinguish between them.
        """
        samples = []
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs["logits"]

            # Good: sample from high-confidence tokens
            good_logits = logits[:, -1, :] / 0.5  # Low temperature = confident
            good_probs = F.softmax(good_logits, dim=-1)
            good_token = torch.multinomial(good_probs, num_samples=1)
            good_sequence = torch.cat([input_ids, good_token], dim=-1)

            # Bad: sample from low-confidence tokens (noise)
            bad_logits = logits[:, -1, :] * -1 + torch.randn_like(logits[:, -1, :]) * 2
            bad_probs = F.softmax(bad_logits, dim=-1)
            bad_token = torch.multinomial(bad_probs, num_samples=1)
            bad_sequence = torch.cat([input_ids, bad_token], dim=-1)

        model.train()

        # Good pair
        samples.append(DreamSample(
            input_ids=good_sequence[:, :-1],
            labels=good_sequence[:, 1:],
            source="contrastive",
            confidence=0.9,
        ))

        # Bad pair (with low confidence label)
        samples.append(DreamSample(
            input_ids=bad_sequence[:, :-1],
            labels=bad_sequence[:, 1:],
            source="contrastive",
            confidence=0.2,
        ))

        return samples

    def generate_knowledge_distillation(
        self,
        input_ids: torch.Tensor,
        model: nn.Module,
        temperature: float = 2.0,
    ) -> DreamSample:
        """
        Knowledge distillation: train on soft labels from own predictions.

        The model's own probability distribution (soft labels) contains
        more information than hard labels - it knows "close but wrong"
        answers are different from "completely wrong" answers.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
            soft_logits = outputs["logits"]
            # Soft targets with temperature
            soft_targets = F.softmax(soft_logits / temperature, dim=-1)

        model.train()

        return DreamSample(
            input_ids=input_ids,
            labels=soft_targets.argmax(dim=-1),  # Hard labels from soft
            source="distillation",
            confidence=0.85,
        )

    def generate_curriculum_batch(
        self,
        batch_size: int,
        seq_len: int,
        difficulty: float = 0.5,
    ) -> DreamSample:
        """
        Generate curriculum learning batch at specified difficulty.

        Easy: short sequences, common tokens
        Medium: medium sequences, mixed tokens
        Hard: long sequences, rare tokens
        """
        # Adjust vocabulary range by difficulty
        vocab_range = int(self.vocab_size * (0.3 + 0.7 * difficulty))
        start_token = max(0, int(self.vocab_size * (1 - difficulty) * 0.5))

        input_ids = torch.randint(
            start_token, start_token + vocab_range,
            (batch_size, seq_len),
        )

        # Labels are shifted input (next token prediction)
        labels = input_ids.clone()

        return DreamSample(
            input_ids=input_ids,
            labels=labels,
            source="curriculum",
            confidence=difficulty,
        )
