"""
Native Interleaving System.

Generates any modality (text, image, video, audio) inline with text.
All modalities share the same token vocabulary and attention mechanism.

Architecture:
  - Unified Token Space: All modalities use the same tokenizer
  - Modality Tokens: Special tokens mark modality boundaries
  - Cross-Modal Attention: Each modality attends to all others
  - Streaming Generation: Generate tokens sequentially
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from enum import Enum


# ==============================================================================
# Modality Types
# ==============================================================================

class Modality(Enum):
    """Modality types for interleaving."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# ==============================================================================
# Special Tokens
# ==============================================================================

SPECIAL_TOKENS = {
    "image_start": 0,
    "image_end": 1,
    "video_start": 2,
    "video_end": 3,
    "audio_start": 4,
    "audio_end": 5,
    "text_start": 6,
    "text_end": 7,
    "pad": 8,
    "unk": 9,
}


# ==============================================================================
# Unified Tokenizer
# ==============================================================================

class UnifiedTokenizer(nn.Module):
    """Unified tokenizer for all modalities.

    Text: BPE tokens
    Image: VAE latent patches
    Video: 3D VAE latent patches
    Audio: Neural codec tokens
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        image_vocab_size: int = 8192,
        video_vocab_size: int = 8192,
        audio_vocab_size: int = 8192,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Text tokens
        self.text_embed = nn.Embedding(vocab_size, embed_dim)

        # Image tokens (from VAE codebook)
        self.image_embed = nn.Embedding(image_vocab_size, embed_dim)

        # Video tokens (from 3D VAE codebook)
        self.video_embed = nn.Embedding(video_vocab_size, embed_dim)

        # Audio tokens (from neural codec codebook)
        self.audio_embed = nn.Embedding(audio_vocab_size, embed_dim)

        # Modality embeddings
        self.modality_embed = nn.Embedding(4, embed_dim)  # 0=text, 1=image, 2=video, 3=audio

    def embed_tokens(
        self,
        tokens: torch.Tensor,
        modality: Modality,
    ) -> torch.Tensor:
        """Embed tokens for a specific modality.

        Args:
            tokens: (B, ...) token IDs
            modality: Modality type

        Returns:
            (B, ..., embed_dim) embeddings
        """
        if modality == Modality.TEXT:
            return self.text_embed(tokens)
        elif modality == Modality.IMAGE:
            return self.image_embed(tokens)
        elif modality == Modality.VIDEO:
            return self.video_embed(tokens)
        elif modality == Modality.AUDIO:
            return self.audio_embed(tokens)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def get_modality_id(self, modality: Modality) -> int:
        """Get modality ID for embedding."""
        return {"text": 0, "image": 1, "video": 2, "audio": 3}[modality.value]


# ==============================================================================
# Interleaved Sequence
# ==============================================================================

class InterleavedSequence:
    """Represents an interleaved sequence of modalities."""

    def __init__(self):
        self.segments: List[Tuple[Modality, torch.Tensor]] = []

    def add_text(self, tokens: torch.Tensor) -> "InterleavedSequence":
        """Add text segment."""
        self.segments.append((Modality.TEXT, tokens))
        return self

    def add_image(self, tokens: torch.Tensor) -> "InterleavedSequence":
        """Add image segment."""
        self.segments.append((Modality.IMAGE, tokens))
        return self

    def add_video(self, tokens: torch.Tensor) -> "InterleavedSequence":
        """Add video segment."""
        self.segments.append((Modality.VIDEO, tokens))
        return self

    def add_audio(self, tokens: torch.Tensor) -> "InterleavedSequence":
        """Add audio segment."""
        self.segments.append((Modality.AUDIO, tokens))
        return self

    def to_tensor(self, tokenizer: UnifiedTokenizer) -> Tuple[torch.Tensor, List[Tuple[int, int, Modality]]]:
        """Convert to tensor with segment metadata.

        Returns:
            tokens: (1, total_len) concatenated token tensor
            segments: List of (start, end, modality) for each segment
        """
        all_tokens = []
        segments = []
        offset = 0

        for modality, tokens in self.segments:
            # Add modality start token
            start_token = SPECIAL_TOKENS[f"{modality.value}_start"]
            all_tokens.append(torch.tensor([start_token]))

            # Add content tokens
            all_tokens.append(tokens)

            # Add modality end token
            end_token = SPECIAL_TOKENS[f"{modality.value}_end"]
            all_tokens.append(torch.tensor([end_token]))

            # Record segment
            start = offset + 1  # Skip start token
            end = offset + 1 + tokens.shape[-1]
            segments.append((start, end, modality))

            offset = end + 1  # +1 for end token

        return torch.cat(all_tokens, dim=-1).unsqueeze(0), segments


# ==============================================================================
# Native Interleaving Generator
# ==============================================================================

class NativeInterleaving(nn.Module):
    """Native interleaving generator for all modalities.

    Generates any modality inline with text using a unified token space.

    Usage:
        generator = NativeInterleaving()

        # Generate interleaved output
        seq = generator.generate(
            "Generate a picture of a cat playing piano, then describe it",
            modalities=["text", "image", "text"],
        )

        # Access generated content
        text_tokens = seq.get_text()
        image_tokens = seq.get_image()
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        image_vocab_size: int = 8192,
        video_vocab_size: int = 8192,
        audio_vocab_size: int = 8192,
        embed_dim: int = 1024,
        hidden_dim: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Unified tokenizer
        self.tokenizer = UnifiedTokenizer(
            vocab_size, image_vocab_size, video_vocab_size, audio_vocab_size, embed_dim
        )

        # Transformer backbone
        self.pos_embed = nn.Embedding(4096, embed_dim)  # Max sequence length

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output heads for each modality
        self.text_head = nn.Linear(embed_dim, vocab_size)
        self.image_head = nn.Linear(embed_dim, image_vocab_size)
        self.video_head = nn.Linear(embed_dim, video_vocab_size)
        self.audio_head = nn.Linear(embed_dim, audio_vocab_size)

        # Modality prediction head
        self.modality_head = nn.Linear(embed_dim, 4)  # Predict which modality to generate

    def _get_modality_mask(self, seq_len: int, current_pos: int, modality: Modality) -> torch.Tensor:
        """Get attention mask for modality-specific generation.

        Args:
            seq_len: Total sequence length
            current_pos: Current position in sequence
            modality: Current modality being generated

        Returns:
            (seq_len,) boolean mask
        """
        mask = torch.ones(seq_len, dtype=torch.bool)

        # Allow attention to all previous positions
        mask[:current_pos] = False

        # Allow attention to current modality's start token
        # (in practice, this is handled by the special tokens)

        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        modality: Modality,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the interleaving generator.

        Args:
            tokens: (B, seq_len) token IDs
            modality: Current modality
            positions: (B, seq_len) position IDs (optional)

        Returns:
            (B, seq_len, vocab_size) logits for next token
        """
        B, seq_len = tokens.shape

        # Embed tokens
        h = self.tokenizer.embed_tokens(tokens, modality)

        # Add position embedding
        if positions is None:
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        h = h + self.pos_embed(positions)

        # Add modality embedding
        mod_id = self.tokenizer.get_modality_id(modality)
        mod_emb = self.tokenizer.modality_embed(torch.tensor(mod_id, device=tokens.device))
        h = h + mod_emb.view(1, 1, -1)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # Output logits
        if modality == Modality.TEXT:
            return self.text_head(h)
        elif modality == Modality.IMAGE:
            return self.image_head(h)
        elif modality == Modality.VIDEO:
            return self.video_head(h)
        elif modality == Modality.AUDIO:
            return self.audio_head(h)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        modalities: List[str],
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> InterleavedSequence:
        """Generate interleaved content.

        Args:
            prompt: Text prompt
            modalities: List of modalities to generate ("text", "image", "video", "audio")
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            InterleavedSequence with generated content
        """
        device = next(self.parameters()).device

        # Start with text prompt
        # In practice, you'd tokenize the prompt
        prompt_tokens = torch.zeros(1, len(prompt.split()), dtype=torch.long, device=device)

        # Generate each modality
        seq = InterleavedSequence()
        seq.add_text(prompt_tokens)

        for modality_str in modalities:
            modality = Modality(modality_str)

            # Generate tokens for this modality
            tokens = self._generate_modality(
                seq, modality, max_length, temperature, top_k, top_p
            )

            # Add to sequence
            if modality == Modality.TEXT:
                seq.add_text(tokens)
            elif modality == Modality.IMAGE:
                seq.add_image(tokens)
            elif modality == Modality.VIDEO:
                seq.add_video(tokens)
            elif modality == Modality.AUDIO:
                seq.add_audio(tokens)

        return seq

    def _generate_modality(
        self,
        context: InterleavedSequence,
        modality: Modality,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Generate tokens for a specific modality.

        Args:
            context: Previous context
            modality: Modality to generate
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated tokens
        """
        device = next(self.parameters()).device

        # Convert context to tensor
        tokens, segments = context.to_tensor(self.tokenizer)
        tokens = tokens.to(device)

        # Get start token for this modality
        start_token = SPECIAL_TOKENS[f"{modality.value}_start"]
        end_token = SPECIAL_TOKENS[f"{modality.value}_end"]

        # Initialize with start token
        generated = torch.tensor([[start_token]], device=device)

        # Generate tokens
        for _ in range(max_length):
            # Forward pass
            logits = self.forward(torch.cat([tokens, generated], dim=1), modality)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, [-1]]] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for end token
            if next_token.item() == end_token:
                break

            # Add to generated
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]  # Remove start token

    def get_text(self, sequence: InterleavedSequence) -> Optional[torch.Tensor]:
        """Extract text tokens from sequence."""
        for modality, tokens in sequence.segments:
            if modality == Modality.TEXT:
                return tokens
        return None

    def get_image(self, sequence: InterleavedSequence) -> Optional[torch.Tensor]:
        """Extract image tokens from sequence."""
        for modality, tokens in sequence.segments:
            if modality == Modality.IMAGE:
                return tokens
        return None

    def get_video(self, sequence: InterleavedSequence) -> Optional[torch.Tensor]:
        """Extract video tokens from sequence."""
        for modality, tokens in sequence.segments:
            if modality == Modality.VIDEO:
                return tokens
        return None

    def get_audio(self, sequence: InterleavedSequence) -> Optional[torch.Tensor]:
        """Extract audio tokens from sequence."""
        for modality, tokens in sequence.segments:
            if modality == Modality.AUDIO:
                return tokens
        return None
