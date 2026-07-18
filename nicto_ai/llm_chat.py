"""
NICTO LLM Chat — Uses NICTO's own model as the brain.

NICTO IS the LLM. No external APIs. No OpenAI. Just NICTO.

Usage:
    # Train first (if not trained yet):
    python -m nicto_ai.training.train --config colab

    # Then chat:
    python -m nicto_ai.llm_chat
"""

import torch
import sys
import os
import json
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class NictoLLM:
    """
    NICTO as an LLM.

    Loads the trained NICTO model and generates text responses.
    NICTO IS the language model.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Find checkpoint
        if checkpoint_path is None:
            candidates = [
                "nicto_model_final.pt",
                "checkpoints/latest.pt",
                "checkpoints/nicto.pt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    checkpoint_path = c
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found. Using untrained model (responses will be random).")
            print("Train with: python -m nicto_ai.training.train --config colab")
            config = NICTOTrainConfig()
            self.model = NICTOTrainModel(config).to(self.device)
            self.model.eval()
            self.config = config

        self.config = self.model.config
        self.vocab_size = self.config.vocab_size

        # Simple byte-level tokenizer
        self.eos_token = 0

    def _load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig

        print(f"Loading NICTO from {path}...")
        raw = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
        config_dict = raw.get("config", {}) if isinstance(raw, dict) else {}

        config = NICTOTrainConfig(
            **{k: v for k, v in config_dict.items() if k in NICTOTrainConfig.__dataclass_fields__}
        )

        model = NICTOTrainModel(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"  {len(missing)} missing keys (random init)")
        if unexpected:
            print(f"  {len(unexpected)} unexpected keys (ignored)")

        self.model = model.to(self.device)
        self.model.eval()
        params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded: {params:,} params, dim={config.dim}, {config.reasoning_layers} reasoning layers")

    def tokenize(self, text: str) -> torch.Tensor:
        """Encode text to token ids."""
        tokens = [b % self.vocab_size for b in text.encode("utf-8", errors="ignore")]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def decode(self, ids: torch.Tensor) -> str:
        """Decode token ids to text."""
        # Convert token ids back to bytes (simple byte-level)
        byte_list = []
        for token_id in ids[0].tolist():
            byte_list.append(token_id % 256)
        return bytes(byte_list).decode("utf-8", errors="replace")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_k: int = 50) -> str:
        """Generate text from a prompt."""
        input_ids = self.tokenize(prompt)

        for _ in range(max_new_tokens):
            # Truncate if too long
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            logits = self.model(input_ids)["logits"][:, -1] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() == self.eos_token:
                break

        # Decode only the new tokens
        new_tokens = input_ids[0, input_ids.shape[1] - max_new_tokens:]
        return self.decode(new_tokens.unsqueeze(0))


class NictoLLMChat:
    """
    Chat with NICTO as the LLM.

    NICTO understands your messages and replies using its own neural network.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.llm = NictoLLM(checkpoint_path=checkpoint_path)
        self.history: List[dict] = []
        self.system_prompt = (
            "You are NICTO AI, a helpful assistant. "
            "You are a neural network with reasoning, memory, emotion, and creativity. "
            "Answer the user's questions helpfully and concisely."
        )

    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        parts = [f"System: {self.system_prompt}"]
        for msg in self.history[-10:]:  # Last 10 messages
            role = msg["role"].title()
            parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def chat(self, user_message: str) -> str:
        """Send a message, get NICTO's reply."""
        self.history.append({"role": "user", "content": user_message})

        prompt = self._build_prompt()
        response = self.llm.generate(prompt, max_new_tokens=256, temperature=0.7)

        # Clean up response
        response = response.strip()
        if response.startswith("Assistant:"):
            response = response[len("Assistant:"):].strip()

        self.history.append({"role": "assistant", "content": response})
        return response

    def clear(self):
        self.history.clear()


def run_llm_chat():
    """Run interactive chat with NICTO as the LLM."""
    print("  _   _ ___ ___  ____ ___  ")
    print(" | \\ | |_ _/ _ \\/ ___/ _ \\ ")
    print(" |  \\| | | | | | |  | | | |")
    print(" | |\\  | | | |_| | |__| |_| |")
    print(" |_| \\_|___|\\___/\\____\\___/ ")
    print()
    print("  NICTO LLM - I AM the language model")
    print("  Type 'quit' to exit, 'clear' to reset")
    print("  " + "=" * 40)
    print()

    chat = NictoLLMChat()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            chat.clear()
            print("(History cleared)\n")
            continue

        response = chat.chat(user_input)
        print(f"NICTO: {response}\n")


if __name__ == "__main__":
    run_llm_chat()
