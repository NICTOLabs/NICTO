"""
NICTO Model Validation
=======================
Validates the final 3B model with generation tests and metrics.
"""
import sys, os, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch

LOG_FILE = "validation.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    log("=" * 60)
    log("NICTO 3B MODEL VALIDATION")
    log("=" * 60)

    from nicto_ai.training.model_master import NICTOMasterModel

    log(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = NICTOMasterModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {params:,} ({params/1e9:.2f}B)")
    log(f"dim={cfg.dim}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    log(f"Training loss: {ckpt.get('loss', '?')}")

    # Test generation with various prompts
    prompts = [
        "The capital of France is",
        "In machine learning, a transformer is",
        "The meaning of life is",
        "def fibonacci(n):",
        "Once upon a time",
        "The three laws of thermodynamics state that",
    ]

    log(f"\nGeneration tests ({len(prompts)} prompts):")
    log("-" * 60)

    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        tokens = _encode_prompt(prompt)
        t0 = time.time()
        with torch.no_grad():
            generated = model.generate(
                tokens, max_new_tokens=50,
                temperature=0.8, top_k=50,
            )
        elapsed = time.time() - t0
        output_text = _decode_tokens(generated[0])

        new_tokens = generated.size(1) - tokens.size(1)
        total_tokens += new_tokens
        total_time += elapsed

        log(f"\nPrompt: \"{prompt}\"")
        log(f"Output: \"{output_text}\"")
        log(f"  {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")

    log(f"\n{'='*60}")
    log(f"Summary:")
    log(f"  Parameters: {params:,} ({params/1e9:.2f}B)")
    log(f"  Total tokens generated: {total_tokens}")
    log(f"  Total time: {total_time:.1f}s")
    log(f"  Avg speed: {total_tokens/total_time:.1f} tok/s")
    log(f"{'='*60}")


def _encode_prompt(text):
    from nicto_ai.tokenizer import get_tokenizer
    tok = get_tokenizer()
    return torch.tensor([tok.encode(text).ids])


def _decode_tokens(token_ids):
    from nicto_ai.tokenizer import get_tokenizer
    tok = get_tokenizer()
    return tok.decode(token_ids.tolist())


if __name__ == "__main__":
    main()
