"""
Creativity Engine.

Handles cross-modal consistency and in-context generation.
Ensures that generated content is consistent across modalities
and builds upon previous outputs.

Architecture:
  - Cross-Modal Consistency: All outputs share the same latent space
  - In-Context Generation: Previous outputs condition next generation
  - Style Transfer: Transfer style between modalities
  - Concept Blending: Combine concepts from different modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from .unified_vae import UnifiedVAE


# ==============================================================================
# Cross-Modal Consistency Module
# ==============================================================================

class CrossModalConsistency(nn.Module):
    """Ensures consistency across modalities.

    When generating a cat in an image and then in a video,
    the cat should look the same in both.
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()

        # Projection networks for each modality
        self.image_proj = nn.Linear(latent_dim, hidden_dim)
        self.video_proj = nn.Linear(latent_dim, hidden_dim)
        self.audio_proj = nn.Linear(latent_dim, hidden_dim)
        self.text_proj = nn.Linear(latent_dim, hidden_dim)

        # Consistency predictor
        self.consistency_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Style projector (shared across modalities)
        self.style_proj = nn.Linear(hidden_dim, 256)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        modality1: str,
        modality2: str,
    ) -> torch.Tensor:
        """Compute consistency score between two latent representations.

        Args:
            z1: (B, latent_dim) first latent
            z2: (B, latent_dim) second latent
            modality1: Modality of first latent
            modality2: Modality of second latent

        Returns:
            (B,) consistency scores in [0, 1]
        """
        # Project to common space
        if modality1 == "image":
            h1 = self.image_proj(z1)
        elif modality1 == "video":
            h1 = self.video_proj(z1)
        elif modality1 == "audio":
            h1 = self.audio_proj(z1)
        elif modality1 == "text":
            h1 = self.text_proj(z1)
        else:
            raise ValueError(f"Unknown modality: {modality1}")

        if modality2 == "image":
            h2 = self.image_proj(z2)
        elif modality2 == "video":
            h2 = self.video_proj(z2)
        elif modality2 == "audio":
            h2 = self.audio_proj(z2)
        elif modality2 == "text":
            h2 = self.text_proj(z2)
        else:
            raise ValueError(f"Unknown modality: {modality2}")

        # Concatenate and predict consistency
        h = torch.cat([h1, h2], dim=-1)
        score = torch.sigmoid(self.consistency_net(h))

        return score.squeeze(-1)

    def compute_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        modality1: str,
        modality2: str,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency loss.

        Args:
            z1, z2: Latent representations
            modality1, modality2: Modalities
            labels: (B,) binary labels (1=consistent, 0=inconsistent)

        Returns:
            Scalar loss
        """
        scores = self.forward(z1, z2, modality1, modality2)
        loss = F.binary_cross_entropy(scores, labels.float())
        return loss

    def get_style(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """Extract style vector from latent.

        Args:
            z: (B, latent_dim) latent vector
            modality: Modality type

        Returns:
            (B, 256) style vector
        """
        if modality == "image":
            h = self.image_proj(z)
        elif modality == "video":
            h = self.video_proj(z)
        elif modality == "audio":
            h = self.audio_proj(z)
        elif modality == "text":
            h = self.text_proj(z)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return self.style_proj(h)


# ==============================================================================
# In-Context Generation Module
# ==============================================================================

class InContextGeneration(nn.Module):
    """Generates new content based on previous outputs.

    Uses attention over previous outputs to condition generation.
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024, num_heads: int = 8):
        super().__init__()

        # Memory bank for previous outputs
        self.memory_proj = nn.Linear(latent_dim, hidden_dim)

        # Cross-attention to memory
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z_current: torch.Tensor,
        memory: List[torch.Tensor],
        modality: str,
    ) -> torch.Tensor:
        """Generate new content conditioned on memory.

        Args:
            z_current: (B, latent_dim) current latent (noise or partial)
            memory: List of (B, latent_dim) previous outputs
            modality: Output modality

        Returns:
            (B, latent_dim) updated latent
        """
        B = z_current.shape[0]
        device = z_current.device

        # Project current latent
        h_current = self.memory_proj(z_current).unsqueeze(1)  # (B, 1, hidden_dim)

        # Project memory
        if len(memory) > 0:
            memory_tensor = torch.stack(memory, dim=1)  # (B, M, latent_dim)
            h_memory = self.memory_proj(memory_tensor)
        else:
            h_memory = torch.zeros(B, 0, self.memory_proj.out_features, device=device)

        # Cross-attention
        h, _ = self.cross_attn(h_current, h_memory, h_memory)
        h = self.norm(h_current + h)

        # Output projection
        h = self.output_proj(h.squeeze(1))

        return z_current + h


# ==============================================================================
# Style Transfer Module
# ==============================================================================

class StyleTransfer(nn.Module):
    """Transfer style between modalities.

    For example, transfer the style of a painting to a video.
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()

        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # AdaIN (Adaptive Instance Normalization)
        self.style_scale = nn.Linear(hidden_dim, hidden_dim)
        self.style_shift = nn.Linear(hidden_dim, hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_content: torch.Tensor,
        z_style: torch.Tensor,
    ) -> torch.Tensor:
        """Transfer style from z_style to z_content.

        Args:
            z_content: (B, latent_dim) content latent
            z_style: (B, latent_dim) style latent

        Returns:
            (B, latent_dim) stylized latent
        """
        # Encode
        h_content = self.content_encoder(z_content)
        h_style = self.style_encoder(z_style)

        # AdaIN
        scale = self.style_scale(h_style)
        shift = self.style_shift(h_style)
        h = h_content * (1 + scale) + shift

        # Decode
        return self.decoder(h)


# ==============================================================================
# Concept Blending Module
# ==============================================================================

class ConceptBlending(nn.Module):
    """Blend concepts from different modalities.

    For example, blend "cat" (text) with "piano" (audio) to generate
    "cat playing piano" (image/video).
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()

        # Concept encoders
        self.concept_encoders = nn.ModuleDict({
            "text": nn.Linear(latent_dim, hidden_dim),
            "image": nn.Linear(latent_dim, hidden_dim),
            "video": nn.Linear(latent_dim, hidden_dim),
            "audio": nn.Linear(latent_dim, hidden_dim),
        })

        # Blending network
        self.blend_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Concept decoder
        self.decoder = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        modality1: str,
        modality2: str,
        blend_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Blend two concepts.

        Args:
            z1: (B, latent_dim) first concept
            z2: (B, latent_dim) second concept
            modality1: Modality of first concept
            modality2: Modality of second concept
            blend_ratio: Ratio of blending (0=z1 only, 1=z2 only)

        Returns:
            (B, latent_dim) blended concept
        """
        # Encode concepts
        h1 = self.concept_encoders[modality1](z1)
        h2 = self.concept_encoders[modality2](z2)

        # Blend
        h = torch.cat([h1, h2], dim=-1)
        h = self.blend_net(h)

        # Weighted blend
        h = (1 - blend_ratio) * h1 + blend_ratio * h

        return self.decoder(h)


# ==============================================================================
# Inspiration Feedback Loop (GAN Validation + Improvement)
# ==============================================================================

class InspirationFeedbackLoop(nn.Module):
    """NICTO's core creativity validation and improvement cycle.

    GAN-based evaluation of what was created:
    1. Discriminator validates quality and alignment
    2. Checks if it reaches user request
    3. If below threshold OR room to be "more better":
       generates inspiration latent that pushes generator
       to create better than before.

    Architecture:
      - Discriminator: Evaluates realism, quality, alignment
      - Inspiration Generator: Produces latent shift direction
        that improves next generation
      - Quality Score: Confidence metric on how good output is
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim

        # Discriminator-style evaluator (processes + validates creation)
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),  # Created + Target
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Quality evaluator (multi-aspect assessment)
        self.quality_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 quality dimensions
            nn.Sigmoid(),
        )

        # Inspiration generator (produces improvement direction)
        self.inspiration_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, hidden_dim),  # Created + Target + Score
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),  # Direction in [-1, 1]
        )

        # When created is "already good but can be better" vs "not reaching request",
        # the inspiration strength is modulated differently
        self.inspiration_strength = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def evaluate(
        self,
        z_created: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """GAN evaluates what was created.

        Uses discriminator to process and validate creation.
        Returns quality score [0, 1] per batch item.

        Args:
            z_created: (B, latent_dim) What was created
            z_target: (B, latent_dim) User request / target

        Returns:
            (B,) quality scores
        """
        # Pool target if it's a sequence
        if z_target.dim() > 2:
            z_target = z_target.mean(dim=-2)

        # Ensure created matches target dimensionality
        if z_created.dim() > 2:
            z_created = z_created.mean(dim=-2) if z_created.dim() == z_target.dim() else z_created.reshape(z_target.shape[0], -1)[:, :self.latent_dim]

        # Discriminator evaluation
        combined = torch.cat([z_created, z_target], dim=-1)
        quality = self.discriminator(combined)
        quality = quality.squeeze(-1)

        return quality

    def evaluate_multi_aspect(
        self,
        z_created: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-aspect quality evaluation.

        Returns scores for:
          - [0]: Realism/Authenticity
          - [1]: Creativity/Novelty
          - [2]: Alignment with user request
          - [3]: Overall quality
        """
        if z_target.dim() > 2:
            z_target = z_target.mean(dim=-2)
        if z_created.dim() > 2:
            z_created = z_created.mean(dim=-2) if z_created.dim() == z_target.dim() else z_created.reshape(z_target.shape[0], -1)[:, :self.latent_dim]

        combined = torch.cat([z_created, z_target], dim=-1)
        aspects = self.quality_net(combined)

        return aspects

    def inspire(
        self,
        z_current: torch.Tensor,
        score: torch.Tensor,
        z_target: torch.Tensor,
        inspiration_multiplier: float = 0.3,
    ) -> torch.Tensor:
        """Generate inspiration to create better than before.

        When score < 1.0 (room for improvement):
        - Produces an inspiration latent shift
        - The shift moves the latent toward better quality
        - Strength is modulated based on how far from target

        When score is high but not perfect:
        - Generates "creative push" to surpass user expectations
        - Makes output "more better than users request"

        Args:
            z_current: (B, latent_dim) Current creation latent
            score: (B,) Quality score [0, 1]
            z_target: (B, latent_dim) Target/user request
            inspiration_multiplier: How strong the inspiration push

        Returns:
            (B, latent_dim) Improved latent (z_current + inspiration)
        """
        if z_target.dim() > 2:
            z_target = z_target.mean(dim=-2)
        if z_current.dim() > 2:
            z_current = z_current.mean(dim=-2) if z_current.dim() == z_target.dim() else z_current.reshape(z_target.shape[0], -1)[:, :self.latent_dim]

        # Concatenate: current creation + target + current score
        score_expanded = score.view(-1, 1)
        combined = torch.cat([z_current, z_target, score_expanded], dim=-1)

        # Generate inspiration direction
        inspiration_direction = self.inspiration_net(combined)

        # Modulate inspiration strength based on score gap
        gap = 1.0 - score  # How far from perfection
        strength = self.inspiration_strength(gap.view(-1, 1)).squeeze(-1)

        # Apply inspiration: push latent toward better quality
        inspiration = inspiration_multiplier * strength.view(-1, 1) * inspiration_direction

        # Inspired latent = current + improvement
        z_inspired = z_current + inspiration

        return z_inspired

    def compute_inspiration_loss(
        self,
        z_created: torch.Tensor,
        z_target: torch.Tensor,
        z_improved: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss to train inspiration loop.

        Loss pushes the discriminator to correctly evaluate
        and the inspiration generator to produce useful improvements.
        """
        # Current quality
        current_score = self.evaluate(z_created, z_target)

        # Improved quality
        improved_score = self.evaluate(z_improved, z_target)

        # We want improved score to be HIGHER than current
        loss = F.mse_loss(improved_score, current_score + 0.1)

        return loss


# ==============================================================================
# Creativity Engine
# ==============================================================================

class CreativityEngine(nn.Module):
    """Main creativity engine combining all modules.

    Handles cross-modal consistency, in-context generation,
    style transfer, and concept blending.

    Usage:
        engine = CreativityEngine()

        # Generate consistent content
        z_image = vae.encode(image, "image")
        z_video = engine.generate_consistent(
            z_image, "video", vae
        )

        # Transfer style
        z_stylized = engine.transfer_style(
            z_image, z_painting
        )

        # Blend concepts
        z_blended = engine.blend_concepts(
            z_cat, z_piano, "text", "audio"
        )
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()

        self.consistency = CrossModalConsistency(latent_dim, hidden_dim)
        self.in_context = InContextGeneration(latent_dim, hidden_dim)
        self.style_transfer = StyleTransfer(latent_dim, hidden_dim)
        self.concept_blending = ConceptBlending(latent_dim, hidden_dim)
        self.inspiration_loop = InspirationFeedbackLoop(latent_dim, hidden_dim)

        # Memory for in-context generation
        self.memory: List[torch.Tensor] = []

    def generate_consistent(
        self,
        z_reference: torch.Tensor,
        target_modality: str,
        source_modality: str,
    ) -> torch.Tensor:
        """Generate content consistent with reference.

        Args:
            z_reference: (B, latent_dim) reference latent
            target_modality: Target modality
            source_modality: Source modality

        Returns:
            (B, latent_dim) consistent latent
        """
        # Add to memory
        self.memory.append(z_reference.detach())

        # Generate using in-context module
        z_current = torch.randn_like(z_reference)
        z_new = self.in_context(z_current, self.memory, target_modality)

        return z_new

    def transfer_style(
        self,
        z_content: torch.Tensor,
        z_style: torch.Tensor,
    ) -> torch.Tensor:
        """Transfer style between modalities.

        Args:
            z_content: (B, latent_dim) content latent
            z_style: (B, latent_dim) style latent

        Returns:
            (B, latent_dim) stylized latent
        """
        return self.style_transfer(z_content, z_style)

    def blend_concepts(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        modality1: str,
        modality2: str,
        blend_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Blend concepts from different modalities.

        Args:
            z1: (B, latent_dim) first concept
            z2: (B, latent_dim) second concept
            modality1: Modality of first concept
            modality2: Modality of second concept
            blend_ratio: Ratio of blending

        Returns:
            (B, latent_dim) blended concept
        """
        return self.concept_blending(z1, z2, modality1, modality2, blend_ratio)

    def check_consistency(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        modality1: str,
        modality2: str,
    ) -> torch.Tensor:
        """Check consistency between two latents.

        Args:
            z1: (B, latent_dim) first latent
            z2: (B, latent_dim) second latent
            modality1: Modality of first latent
            modality2: Modality of second latent

        Returns:
            (B,) consistency scores in [0, 1]
        """
        return self.consistency(z1, z2, modality1, modality2)

    def validate_and_inspire(
        self,
        z_created: torch.Tensor,
        z_target: torch.Tensor,
        modality: str = "image",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAN-based validation and inspiration loop.

        Uses discriminator-style evaluation to check what was created.
        If it doesn't reach user request or can be made better,
        generates inspiration to create better than before.

        Returns:
            z_inspired: (B, latent_dim) inspired latent
            score: (B,) quality score [0, 1]
        """
        score = self.inspiration_loop.evaluate(z_created, z_target)
        z_inspired = self.inspiration_loop.inspire(z_created, score, z_target)
        return z_inspired, score

    def iterate_refinement(
        self,
        generator_fn,
        initial_latent: torch.Tensor,
        z_target: torch.Tensor,
        modality: str = "image",
        max_iterations: int = 5,
        target_score: float = 0.95,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Full GAN validation + inspiration refinement loop.

        1. Generate using generator_fn(latent)
        2. GAN evaluates quality and alignment
        3. If score < target: inspire better latent
        4. Repeat until threshold met or max iterations.

        Returns:
            best_output: Final best generation
            scores: List of scores per iteration
        """
        scores = []
        z = initial_latent

        for iteration in range(max_iterations):
            # Generate
            output = generator_fn(z)

            # GAN validates
            score = self.inspiration_loop.evaluate(z, z_target)
            scores.append(score.mean().item())

            # Check: reached user request or better?
            if score.mean().item() >= target_score:
                break

            # Inspire better than before
            z = self.inspiration_loop.inspire(z, score, z_target)

        return output, scores

    def clear_memory(self):
        """Clear the memory bank."""
        self.memory = []
