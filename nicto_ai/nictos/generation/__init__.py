"""
NICTO Unified Generation Engine.

Unified multimodal generation system that surpasses Gemini Omni's capabilities.
All modalities (text, image, video, audio) share a common latent space
and are processed by a single DiT backbone with flow matching.

Architecture:
  1. Unified VAE: encode all modalities to shared latent space
  2. Flow Matching DiT: denoise latents conditioned on all modalities
  3. Native Interleaving: generate any modality inline with text
  4. Cross-Modal Consistency: outputs are visually/audibly consistent
"""

from .unified_vae import UnifiedVAE
from .flow_matching import DiT, FlowMatchingTrainer, FlowMatchingSampler
from .video_generator import VideoGenerator
from .audio_generator import MusicGenerator, SFXGenerator
from .native_interleaving import NativeInterleaving
from .creativity_engine import CreativityEngine, InspirationFeedbackLoop
