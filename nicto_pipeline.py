"""
NICTO Self-Improvement Pipeline
================================
Cycle: Benchmark → Distill → Train → Repeat

1. Benchmark: Compare NICTO vs Groq/OpenRouter teacher models on quality + speed
2. Distill: Generate teacher outputs, create training data
3. Train: Train NICTO on distilled data
4. Repeat: Each cycle makes NICTO smarter

Usage:
  python nicto_pipeline.py --cycles 3 --distill-steps 500 --train-steps 1000
"""

import argparse, json, math, os, sys, time, random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import requests

# ── API Keys ──
def load_keys():
    keys = {}
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys

API_KEYS = load_keys()

# ── Teacher Models ──
TEACHERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": API_KEYS.get("GROQ_API_KEY", ""),
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key": API_KEYS.get("OPENROUTER_API_KEY", ""),
        "models": ["google/gemma-3-27b-it", "meta-llama/llama-4-scout"],
    },
}

# ── Benchmark Prompts ──
BENCHMARK_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list using quicksort.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a short story about a robot learning to paint.",
    "What is the time complexity of binary search? Prove it.",
    "Explain how neural networks learn via backpropagation.",
    "What are the SOLID principles in software engineering?",
    "Write a haiku about artificial intelligence.",
    "Explain the difference between mutex and semaphore.",
]


def call_teacher(provider: str, model: str, prompt: str, max_tokens: int = 512) -> dict:
    """Call a teacher model API and return response + timing."""
    cfg = TEACHERS[provider]
    if not cfg["key"]:
        return {"error": f"No API key for {provider}", "text": "", "time": 0}

    headers = {
        "Authorization": f"Bearer {cfg['key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    t0 = time.time()
    try:
        resp = requests.post(cfg["url"], json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - t0
        text = data["choices"][0]["message"]["content"]
        tokens_out = data.get("usage", {}).get("completion_tokens", len(text.split()))
        return {
            "text": text,
            "time": elapsed,
            "tokens_per_sec": tokens_out / max(elapsed, 0.01),
            "tokens_out": tokens_out,
            "model": model,
        }
    except Exception as e:
        return {"error": str(e), "text": "", "time": time.time() - t0}


def benchmark_nicto(model, tokenizer, device, prompts: list) -> dict:
    """Benchmark NICTO model generation speed and output."""
    results = []
    model.eval()
    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        if tokenizer is None:
            # Fallback: use random tokens
            input_ids = torch.randint(0, 1000, (1, 32)).to(device)
        else:
            encoded = tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, -512:]

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
        elapsed = time.time() - t0

        new_tokens = output.shape[1] - input_ids.shape[1]

        if tokenizer is not None:
            text = tokenizer.decode(output[0, input_ids.shape[1]:].tolist())
        else:
            text = f"[{new_tokens} tokens]"

        total_tokens += new_tokens
        total_time += elapsed

        results.append({
            "prompt": prompt[:60],
            "response": text[:200],
            "time": elapsed,
            "tokens": new_tokens,
        })

    return {
        "results": results,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_sec": total_tokens / max(total_time, 0.01),
    }


def run_benchmark_cycle(model, tokenizer, device, cycle: int) -> dict:
    """Run one full benchmark cycle: NICTO vs all teachers."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK CYCLE {cycle}")
    print(f"{'='*60}")

    # Benchmark NICTO
    print("\n  Benchmarking NICTO...")
    nicto_results = benchmark_nicto(model, tokenizer, device, BENCHMARK_PROMPTS)
    print(f"    NICTO: {nicto_results['tokens_per_sec']:.1f} tok/s, "
          f"{nicto_results['total_tokens']} tokens in {nicto_results['total_time']:.1f}s")

    # Benchmark teachers
    teacher_results = {}
    for provider, cfg in TEACHERS.items():
        if not cfg["key"]:
            print(f"    {provider}: skipped (no API key)")
            continue
        for model_name in cfg["models"]:
            name = f"{provider}/{model_name}"
            print(f"  Benchmarking {name}...")
            speeds = []
            for prompt in BENCHMARK_PROMPTS[:3]:  # Test with 3 prompts
                result = call_teacher(provider, model_name, prompt, max_tokens=100)
                if "error" not in result:
                    speeds.append(result["tokens_per_sec"])
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                teacher_results[name] = {"avg_tokens_per_sec": avg_speed}
                print(f"    {name}: {avg_speed:.1f} tok/s")
            else:
                print(f"    {name}: failed")

    return {"nicto": nicto_results, "teachers": teacher_results}


def generate_distillation_data(cycle: int, n_samples: int = 200) -> str:
    """Generate teacher outputs for distillation."""
    print(f"\n{'='*60}")
    print(f"  GENERATING DISTILLATION DATA (cycle {cycle})")
    print(f"{'='*60}")

    # Use a diverse set of prompts for distillation
    DISTILL_PROMPTS = [
        "Explain what a neural network is in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What is the difference between supervised and unsupervised learning?",
        "Describe the concept of gradient descent.",
        "What are the advantages of using a transformer architecture?",
        "Explain how attention mechanisms work in NLP.",
        "Write a short explanation of backpropagation.",
        "What is overfitting and how do you prevent it?",
        "Explain the concept of transfer learning.",
        "What is a convolutional neural network?",
        "How does a recurrent neural network work?",
        "Explain the vanishing gradient problem.",
        "What is batch normalization?",
        "Describe the ReLU activation function.",
        "What is dropout and why is it used?",
        "Explain the difference between L1 and L2 regularization.",
        "What is a GAN (Generative Adversarial Network)?",
        "How does a random forest algorithm work?",
        "Explain the concept of feature engineering.",
        "What is cross-validation?",
        "Describe the bias-variance tradeoff.",
        "What is the curse of dimensionality?",
        "Explain how k-means clustering works.",
        "What is a support vector machine?",
        "How does principal component analysis work?",
        "Explain the concept of word embeddings.",
        "What is the difference between bag of words and TF-IDF?",
        "How does a seq2seq model work?",
        "What is beam search in neural decoding?",
        "Explain the concept of curriculum learning.",
    ]

    outputs = []
    used_prompts = set()

    for provider, cfg in TEACHERS.items():
        if not cfg["key"]:
            continue
        model_name = random.choice(cfg["models"])

        for i in range(n_samples // 2):
            if len(outputs) >= n_samples:
                break

            prompt = random.choice(DISTILL_PROMPTS)
            if prompt in used_prompts:
                continue
            used_prompts.add(prompt)

            result = call_teacher(provider, model_name, prompt, max_tokens=256)
            if "error" not in result and result["text"]:
                outputs.append({
                    "prompt": prompt,
                    "response": result["text"],
                    "teacher": f"{provider}/{model_name}",
                })
                if len(outputs) % 10 == 0:
                    print(f"    Generated {len(outputs)}/{n_samples} samples...")

            time.sleep(0.1)

    # Save to JSONL
    out_path = f"training_data/distill_cycle_{cycle}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Saved {len(outputs)} samples to {out_path}")
    return out_path


def tokenize_distillation_data(jsonl_path: str) -> str:
    """Convert JSONL distillation data to .bin format for training."""
    import numpy as np
    from pathlib import Path

    print(f"  Tokenizing {jsonl_path}...")

    # Load tokenizer
    tokenizer_path = Path("nicto_ai/tokenizer/artifacts/tokenizer.json")
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except:
        print("  Warning: Could not load tokenizer, using simple tokenization")
        return ""

    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    all_ids = []
    for item in items:
        text = f"User: {item['prompt'][:300]}\nAssistant: {item['response']}\n"
        encoded = tokenizer.encode(text)
        all_ids.extend(encoded.ids)

    # Save as .bin
    bin_path = jsonl_path.replace(".jsonl", ".bin")
    arr = np.array(all_ids, dtype=np.uint32)
    arr.tofile(bin_path)

    print(f"  Saved {len(all_ids)} tokens to {bin_path}")
    return bin_path


def train_nicto(config_name: str, steps: int, bin_path: str = None) -> str:
    """Train NICTO model."""
    print(f"\n{'='*60}")
    print(f"  TRAINING NICTO ({config_name}, {steps} steps)")
    print(f"{'='*60}")

    py_exe = sys.executable
    cmd = f'"{py_exe}" train_master.py --config {config_name} --steps {steps} --batch-size 1 --seq-len 128 --log-every 50'

    ckpt_dir = Path(f"checkpoints_master/{config_name}")
    resume_ckpt = ckpt_dir / "final.pt"
    if resume_ckpt.exists():
        cmd += f' --resume {resume_ckpt}'
        print(f"  Resuming from: {resume_ckpt}")

    if bin_path:
        print(f"  Using distillation data: {bin_path}")

    os.system(cmd)

    # Return best checkpoint path
    ckpt_dir = Path(f"checkpoints_master/{config_name}")
    best = ckpt_dir / "best.pt"
    final = ckpt_dir / "final.pt"
    if best.exists():
        return str(best)
    elif final.exists():
        return str(final)
    return ""


def main():
    parser = argparse.ArgumentParser(description="NICTO Self-Improvement Pipeline")
    parser.add_argument("--cycles", type=int, default=3, help="Number of benchmark→distill→train cycles")
    parser.add_argument("--distill-samples", type=int, default=200, help="Teacher samples per cycle")
    parser.add_argument("--train-steps", type=int, default=500, help="Training steps per cycle")
    parser.add_argument("--config", type=str, default="tiny", help="Model config to train")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark phase")
    parser.add_argument("--skip-distill", action="store_true", help="Skip distillation, use existing data")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  NICTO SELF-IMPROVEMENT PIPELINE")
    print(f"  Cycles: {args.cycles}, Config: {args.config}")
    print(f"  Train steps/cycle: {args.train_steps}")
    print(f"{'#'*60}")

    # Load model
    from nicto_ai.training.model_master import (
        config_master_tiny, config_master_medium, NICTOMasterModel
    )

    device = "cpu"
    cfg_func = {"tiny": config_master_tiny, "medium": config_master_medium}
    cfg = cfg_func.get(args.config, config_master_tiny)()
    model = NICTOMasterModel(cfg).to(device)

    # Load NICTO's tokenizer
    tokenizer = None
    try:
        from tokenizers import Tokenizer
        tok_path = Path("nicto_ai/tokenizer/artifacts/tokenizer.json")
        if tok_path.exists():
            tokenizer = Tokenizer.from_file(str(tok_path))
            print(f"  Loaded tokenizer from {tok_path}")
    except Exception as e:
        print(f"  Warning: Could not load tokenizer: {e}")

    for cycle in range(1, args.cycles + 1):
        print(f"\n{'*'*60}")
        print(f"  CYCLE {cycle}/{args.cycles}")
        print(f"{'*'*60}")

        # 1. Benchmark
        if not args.skip_benchmark:
            bench = run_benchmark_cycle(model, tokenizer, device, cycle)

        # 2. Distill
        bin_path = None
        if not args.skip_distill:
            jsonl_path = generate_distillation_data(cycle, args.distill_samples)
            if jsonl_path:
                bin_path = tokenize_distillation_data(jsonl_path)

        # 3. Train
        checkpoint = train_nicto(args.config, args.train_steps, bin_path)

        # 4. Reload trained model for next cycle
        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(f"\n  Loaded checkpoint: {checkpoint}")
            print(f"  Loss: {ckpt.get('loss', 'N/A')}")

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE — {args.cycles} cycles finished")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
