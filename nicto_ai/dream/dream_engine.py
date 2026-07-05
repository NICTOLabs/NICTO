"""
NICTO AI - Dream Engine
Offline learning through experience replay, consolidation, and generation

This is how NICTO "dreams" - during idle periods, it:
1. Replays past experiences (prioritized by surprise/mistakes)
2. Consolidates memories (merges similar, strengthens important)
3. Generates synthetic training data (augmentation, chains, contrastive)
4. Self-evaluates and improves (metacognitive feedback loop)

The dream engine runs offline between training sessions to continuously
improve without requiring new external data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .replay import ExperienceReplayBuffer, Experience
from .generator import SyntheticDataGenerator, DreamSample

logger = logging.getLogger(__name__)


@dataclass
class DreamSession:
    """Results from a single dream session"""
    duration_ms: float = 0.0
    experiences_replayed: int = 0
    samples_generated: int = 0
    avg_loss_before: float = 0.0
    avg_loss_after: float = 0.0
    loss_improvement: float = 0.0
    consolidation_stats: Dict = None

    def __post_init__(self):
        if self.consolidation_stats is None:
            self.consolidation_stats = {}


class DreamEngine:
    """
    NICTO's Dream Engine - offline learning system.

    Runs during idle periods to:
    1. Replay and learn from past experiences
    2. Generate synthetic training data
    3. Consolidate and organize knowledge
    4. Self-evaluate and calibrate

    Unlike standard training (which needs new data), the dream engine
    improves NICTO by reprocessing what it already knows, finding
    patterns it missed, and generating new training signal.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_capacity: int = 50_000,
        vocab_size: int = 32000,
        dim: int = 256,
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_capacity)
        self.generator = SyntheticDataGenerator(vocab_size=vocab_size, dim=dim)
        self._session_count = 0

    def record_experience(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        loss: float,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Record an experience for later replay.

        Called after each training step or inference to capture
        what the model did and how surprised it was.
        """
        with torch.no_grad():
            # Compute surprise (entropy of predictions)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1).mean().item()

            # Check if prediction was correct
            if labels is not None:
                pred_tokens = logits[:, -1, :].argmax(dim=-1)
                actual_tokens = labels[:, -1]
                was_correct = (pred_tokens == actual_tokens).float().mean().item()
            else:
                was_correct = 0.5
                actual_tokens = logits[:, -1, :].argmax(dim=-1)

            # Surprise = how unexpected was this loss given recent history
            avg_loss = self.replay_buffer._stats.get("avg_loss", loss)
            surprise = abs(loss - avg_loss) / max(avg_loss, 1e-6)

            experience = Experience(
                id=self.replay_buffer._id_counter,
                input_ids=input_ids.detach(),
                logits=logits.detach(),
                loss=loss,
                uncertainty=1.0 - was_correct,
                entropy=entropy,
                surprise=surprise,
                predicted_token=logits[:, -1, :].argmax(dim=-1)[0].item(),
                actual_token=actual_tokens[0].item() if actual_tokens.dim() > 0 else actual_tokens.item(),
                was_correct=was_correct > 0.5,
                timestamp=int(time.time()),
            )

            self.replay_buffer.add(experience)

            # Update running stats
            n = self.replay_buffer._stats["total_added"]
            self.replay_buffer._stats["avg_loss"] = (avg_loss * n + loss) / (n + 1)
            self.replay_buffer._stats["avg_surprise"] = (
                self.replay_buffer._stats["avg_surprise"] * n + surprise
            ) / (n + 1)

    def dream(
        self,
        n_replay_steps: int = 100,
        n_generated_batches: int = 10,
        batch_size: int = 4,
        seq_len: int = 32,
        consolidation_interval: int = 50,
    ) -> DreamSession:
        """
        Run a dream session - offline learning from experiences.

        This is the core of NICTO's offline learning:
        1. Sample surprising/wrong experiences from replay buffer
        2. Re-train on those experiences (learn from mistakes)
        3. Generate synthetic data (augmentation, chains, contrastive)
        4. Train on synthetic data (expand knowledge)
        5. Consolidate (organize and strengthen important patterns)

        Args:
            n_replay_steps: How many batches to replay from buffer
            n_generated_batches: How many synthetic batches to generate
            batch_size: Batch size for each step
            seq_len: Sequence length for generated data
            consolidation_interval: Steps between consolidation

        Returns:
            DreamSession with statistics
        """
        start_time = time.time()
        session = DreamSession()
        self._session_count += 1
        self.model.train()

        losses_before = []
        losses_after = []

        # Phase 1: Experience Replay (learn from mistakes)
        logger.info("Dream session %d: Phase 1 - Experience Replay", self._session_count)
        for step in range(n_replay_steps):
            if self.replay_buffer.size < batch_size:
                break

            experiences = self.replay_buffer.sample(batch_size)
            if not experiences:
                break

            # Batch experiences - flatten to [total_tokens] then reshape
            flat_inputs = []
            for e in experiences:
                ids = e.input_ids.flatten()[:seq_len]
                if ids.shape[0] < seq_len:
                    ids = torch.nn.functional.pad(ids, (0, seq_len - ids.shape[0]))
                flat_inputs.append(ids)
            batch_input = torch.stack(flat_inputs)  # [batch_size, seq_len]
            batch_labels = batch_input.clone()

            # Forward + backward
            self.optimizer.zero_grad()
            outputs = self.model(batch_input, labels=batch_labels)
            loss = outputs["loss"]

            losses_before.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses_after.append(outputs["loss"].item())

            session.experiences_replayed += 1

            # Periodic consolidation
            if step > 0 and step % consolidation_interval == 0:
                self._consolidate()

        # Phase 2: Synthetic Data Generation (expand knowledge)
        logger.info("Dream session %d: Phase 2 - Synthetic Generation", self._session_count)
        for step in range(n_generated_batches):
            # Generate different types of synthetic data
            if step % 4 == 0:
                # Augmentation
                dummy_input = torch.randint(0, self.generator.vocab_size, (batch_size, seq_len))
                dummy_logits = torch.randn(batch_size, seq_len, self.generator.vocab_size)
                aug_samples = self.generator.augment_experience(dummy_input, dummy_logits, n_augments=1)
                sample = aug_samples[0]
            elif step % 4 == 1:
                # Curriculum
                difficulty = (step + 1) / n_generated_batches
                sample = self.generator.generate_curriculum_batch(batch_size, seq_len, difficulty)
            elif step % 4 == 2:
                # Contrastive pairs
                dummy_input = torch.randint(0, self.generator.vocab_size, (1, seq_len))
                pairs = self.generator.generate_contrastive_pairs(dummy_input, self.model, n_pairs=1)
                sample = pairs[0]
            else:
                # Knowledge distillation
                dummy_input = torch.randint(0, self.generator.vocab_size, (1, seq_len))
                sample = self.generator.generate_knowledge_distillation(dummy_input, self.model)

            # Train on synthetic sample
            self.optimizer.zero_grad()
            outputs = self.model(sample.input_ids, labels=sample.labels)
            loss = outputs["loss"] * sample.weight if hasattr(sample, 'weight') else outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            session.samples_generated += 1

        # Compute improvement
        if losses_before and losses_after:
            session.avg_loss_before = sum(losses_before) / len(losses_before)
            session.avg_loss_after = sum(losses_after) / len(losses_after)
            session.loss_improvement = session.avg_loss_before - session.avg_loss_after

        session.duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "Dream session %d complete: replayed=%d, generated=%d, loss_improvement=%.4f, time=%.1fs",
            self._session_count,
            session.experiences_replayed,
            session.samples_generated,
            session.loss_improvement,
            session.duration_ms / 1000,
        )

        return session

    def _consolidate(self):
        """
        Consolidate knowledge: re-evaluate stored experiences.

        Re-computes priorities for stored experiences based on
        current model state. Experiences that are still surprising
        after re-evaluation get higher priority.
        """
        if self.replay_buffer.size < 10:
            return

        # Sample some experiences and re-evaluate
        sample_size = min(50, self.replay_buffer.size)
        experiences = self.replay_buffer.sample(sample_size)

        new_losses = []
        for exp in experiences:
            try:
                new_loss = self.replay_buffer.compute_td_error(self.model, exp)
                new_losses.append(new_loss)
            except Exception:
                new_losses.append(exp.loss)

        self.replay_buffer.update_priorities(experiences, new_losses)

    def get_stats(self) -> Dict:
        """Get dream engine statistics."""
        return {
            "session_count": self._session_count,
            "replay_buffer_size": self.replay_buffer.size,
            "replay_stats": self.replay_buffer.stats,
        }
