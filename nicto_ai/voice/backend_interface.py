"""
NICTO AI - LLM Backend Interface
Abstract interface for LLM backends.

The agent_loop.py uses this interface to communicate with
LLM providers without being coupled to any specific API.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """A chat message"""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Result from LLM completion"""
    text: str
    model: str = ""
    usage: Dict = field(default_factory=dict)  # tokens in/out
    finish_reason: str = "stop"
    metadata: Dict = field(default_factory=dict)


class LLMBackend(ABC):
    """
    Abstract LLM backend interface.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> CompletionResult:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class NICTOBackend(LLMBackend):
    """
    NICTO's own model backend.
    Loads a trained NICTOTrainModel checkpoint and generates locally.
    """

    def __init__(self, model=None, checkpoint_path: Optional[str] = None):
        self._model = None
        self._device = None
        self._config = None

        if model is not None:
            self._model = model
        elif checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load NICTO model from checkpoint file."""
        import torch
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig

        self._device = torch.device("cpu")

        try:
            raw = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

            # Handle different checkpoint formats
            state_dict = None
            config_dict = {}

            if isinstance(raw, dict):
                if "model" in raw and isinstance(raw["model"], dict):
                    # New format: {"model": state_dict, "config": {...}, "step": N}
                    state_dict = raw["model"]
                    config_dict = raw.get("config", {})
                elif "tok_emb.weight" in raw:
                    # Old format: raw state_dict at top level
                    state_dict = raw
                    config_dict = {}
                else:
                    state_dict = raw

            if state_dict is None:
                print("NICTOBackend: could not find model weights in checkpoint")
                return

            # Infer config from state_dict if not provided
            if not config_dict:
                config_dict = self._infer_config(state_dict)

            config = NICTOTrainConfig(
                **{k: v for k, v in config_dict.items() if k in NICTOTrainConfig.__dataclass_fields__}
            )
            self._config = config

            # Detect old checkpoint format (single reasoning block vs multi-layer)
            is_old_format = "r_attn.wqkv.weight" in state_dict

            if is_old_format:
                # Remap old keys to new multi-layer format
                state_dict = self._remap_old_keys(state_dict, config)

            model = NICTOTrainModel(config)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"NICTOBackend: {len(missing)} missing keys (will use random init for those)")
            if unexpected:
                print(f"NICTOBackend: {len(unexpected)} unexpected keys (ignored)")

            self._model = model
            self._model.to(self._device)
            self._model.eval()

            params = self._model.count_parameters()
            print(f"NICTOBackend: loaded checkpoint ({params:,} params, dim={config.dim}, reasoning_layers={config.reasoning_layers})")

        except Exception as e:
            print(f"NICTOBackend: failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            self._model = None

    def _infer_config(self, state_dict):
        """Infer NICTOTrainConfig from state_dict keys."""
        # Count memory layers
        mem_keys = [k for k in state_dict if k.startswith("m_layers.") and ".self_attn." in k]
        mem_layers = max([int(k.split(".")[1]) for k in mem_keys], default=0) + 1 if mem_keys else 4

        # Count emotional layers
        emo_keys = [k for k in state_dict if k.startswith("e_layers.") and ".self_attn." in k]
        emo_layers = max([int(k.split(".")[1]) for k in emo_keys], default=0) + 1 if emo_keys else 4

        # Count creative layers
        cre_keys = [k for k in state_dict if k.startswith("c_layers.") and ".self_attn." in k]
        cre_layers = max([int(k.split(".")[1]) for k in cre_keys], default=0) + 1 if cre_keys else 4

        # Detect if old format (single reasoning block)
        is_old = "r_attn.wqkv.weight" in state_dict
        reasoning_layers = 1 if is_old else 6

        # Detect dim from tok_emb
        dim = state_dict["tok_emb.weight"].shape[1] if "tok_emb.weight" in state_dict else 1024

        # Detect MoE experts
        moe_experts = 4
        for k in state_dict:
            if k.startswith("r_moe.experts.") and ".0.0.weight" in k:
                idx = int(k.split(".")[2])
                moe_experts = max(moe_experts, idx + 1)
        if is_old:
            for k in state_dict:
                if k.startswith("r_moe.experts.") and ".0.0.weight" in k:
                    idx = int(k.split(".")[2])
                    moe_experts = max(moe_experts, idx + 1)

        return {
            "vocab_size": 32000,
            "dim": dim,
            "max_seq_len": 1024,
            "reasoning_layers": reasoning_layers,
            "n_heads": 8,
            "n_kv_heads": 2,
            "moe_experts": moe_experts,
            "moe_activated": 2,
            "moe_hidden": dim * 2,
            "memory_layers": mem_layers,
            "emotional_layers": emo_layers,
            "creative_layers": cre_layers,
        }

    def _remap_old_keys(self, state_dict, config):
        """Remap old single-block reasoning keys to new multi-layer format."""
        new_dict = {}

        for key, value in state_dict.items():
            if key.startswith("r_attn."):
                # r_attn.xxx -> r_attn_layers.0.xxx
                new_key = key.replace("r_attn.", "r_attn_layers.0.")
                new_dict[new_key] = value
            elif key.startswith("r_moe."):
                # r_moe.xxx -> r_moe_layers.0.xxx
                new_key = key.replace("r_moe.", "r_moe_layers.0.")
                new_dict[new_key] = value
            else:
                new_dict[key] = value

        # For reasoning_layers > 1, duplicate the single block
        if config.reasoning_layers > 1:
            for layer_idx in range(1, config.reasoning_layers):
                for key in list(new_dict.keys()):
                    if key.startswith("r_attn_layers.0."):
                        new_key = key.replace("r_attn_layers.0.", f"r_attn_layers.{layer_idx}.")
                        new_dict[new_key] = new_dict[key]
                    elif key.startswith("r_moe_layers.0."):
                        new_key = key.replace("r_moe_layers.0.", f"r_moe_layers.{layer_idx}.")
                        new_dict[new_key] = new_dict[key]

        return new_dict

    def _build_prompt(self, messages: List[Message], system_prompt: Optional[str] = None) -> str:
        """Build a text prompt from messages."""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for msg in messages:
            if msg.role in ("user", "assistant"):
                parts.append(f"{msg.role.title()}: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def complete(self, messages, temperature=0.7, max_tokens=128, system_prompt=None, **kwargs):
        if self._model is None:
            return CompletionResult(
                text="[NICTO model not loaded. Provide a checkpoint path.]",
                finish_reason="error",
            )

        import torch

        prompt = self._build_prompt(messages, system_prompt)

        # Tokenize using byte-level encoding (same as training)
        config = self._model.config
        tokens = [b % config.vocab_size for b in prompt.encode("utf-8")]

        if not tokens:
            return CompletionResult(text="", finish_reason="empty")

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)

        # Generate
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
            )

        # Decode only new tokens
        new_tokens = output[0][input_ids.shape[1]:].cpu().tolist()

        # Convert tokens back to text (byte-level decoding)
        text_bytes = bytes([t % 256 for t in new_tokens])
        try:
            text = text_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = str(new_tokens)

        # Clean up - remove garbage after natural stop
        for stop_token in ["\n\n", "Assistant:", "User:", "System:"]:
            idx = text.find(stop_token)
            if idx > 0:
                text = text[:idx].strip()
                break

        return CompletionResult(
            text=text.strip(),
            model=self.name,
            usage={
                "prompt_tokens": len(tokens),
                "completion_tokens": len(new_tokens),
            },
        )

    def is_available(self):
        return self._model is not None

    @property
    def name(self):
        return "nicto"


class EchoBackend(LLMBackend):
    """Simple echo backend for testing the pipeline"""

    def complete(self, messages, **kwargs):
        if messages:
            last = messages[-1]
            return CompletionResult(text=f"Echo: {last.content}")
        return CompletionResult(text="")

    def is_available(self):
        return True

    @property
    def name(self):
        return "echo"
