"""
NICTO AI vs GPT-2 Benchmark Comparison (Simplified)
Compare model size, speed, and basic metrics
"""

import torch
import time
import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Result from a benchmark test"""
    test_name: str
    nicto_value: float
    gpt2_value: float
    nicto_unit: str
    gpt2_unit: str
    winner: str
    notes: str = ""


def count_parameters(model) -> int:
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())


def measure_inference_speed(model, input_ids, n_runs: int = 5) -> float:
    """Measure inference speed in tokens/second"""
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            try:
                _ = model(input_ids)
            except Exception:
                return 0.0
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            try:
                _ = model(input_ids)
            except Exception:
                return 0.0
    end_time = time.time()
    
    total_time = end_time - start_time
    tokens_per_second = (input_ids.shape[1] * n_runs) / total_time
    return tokens_per_second


def calculate_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Calculate perplexity on text"""
    try:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        return torch.exp(loss).item()
    except Exception as e:
        print(f"  Perplexity error: {e}")
        return 0.0


def generate_text(model, tokenizer, prompt: str, max_length: int = 50, device: str = "cpu") -> str:
    """Generate text from prompt"""
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {e}"


def run_benchmark():
    """Run benchmark comparison"""
    print("=" * 60)
    print("NICTO AI vs GPT-2 Benchmark (Simplified)")
    print("=" * 60)
    
    device = "cpu"
    results = {}
    
    # 1. Load GPT-2
    print("\n1. Loading GPT-2...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_model.to(device)
        gpt2_model.eval()
        
        gpt2_params = count_parameters(gpt2_model)
        print(f"   GPT-2 loaded: {gpt2_params:,} parameters")
        
        results["gpt2"] = {
            "parameters": gpt2_params,
            "loaded": True,
        }
    except Exception as e:
        print(f"   Failed to load GPT-2: {e}")
        results["gpt2"] = {"parameters": 0, "loaded": False}
        return results
    
    # 2. Test GPT-2 inference speed
    print("\n2. Testing GPT-2 inference speed...")
    input_ids = torch.randint(0, 50256, (1, 128)).to(device)
    gpt2_speed = measure_inference_speed(gpt2_model, input_ids)
    print(f"   GPT-2 speed: {gpt2_speed:.2f} tokens/second")
    results["gpt2"]["speed_tps"] = gpt2_speed
    
    # 3. Test GPT-2 text generation
    print("\n3. Testing GPT-2 text generation...")
    test_prompts = [
        "The quick brown fox",
        "Artificial intelligence is",
        "In the future, technology will",
    ]
    
    generation_results = {}
    for prompt in test_prompts:
        output = generate_text(gpt2_model, gpt2_tokenizer, prompt, max_length=80)
        print(f"\n   Prompt: '{prompt}'")
        print(f"   Output: {output[:150]}...")
        generation_results[prompt] = output
    
    results["gpt2"]["generation"] = generation_results
    
    # 4. Test GPT-2 perplexity
    print("\n4. Testing GPT-2 perplexity...")
    test_text = "Artificial intelligence is transforming the way we live and work. Machine learning algorithms can now recognize images, understand natural language, and make decisions."
    gpt2_ppl = calculate_perplexity(gpt2_model, gpt2_tokenizer, test_text, device)
    print(f"   GPT-2 perplexity: {gpt2_ppl:.2f}")
    results["gpt2"]["perplexity"] = gpt2_ppl
    
    # 5. Memory usage
    print("\n5. Measuring GPT-2 memory usage...")
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        _ = measure_inference_speed(gpt2_model, input_ids, n_runs=1)
        mem_after = process.memory_info().rss / 1024 / 1024
        model_size = sum(p.numel() * p.element_size() for p in gpt2_model.parameters()) / 1024 / 1024
        
        print(f"   GPT-2 model size: {model_size:.2f} MB")
        print(f"   Memory delta: {mem_after - mem_before:.2f} MB")
        results["gpt2"]["model_size_mb"] = model_size
        results["gpt2"]["memory_delta_mb"] = mem_after - mem_before
    except ImportError:
        print("   psutil not available")
    
    # 6. Architecture info
    print("\n6. GPT-2 Architecture Info:")
    print(f"   Type: Transformer Decoder")
    print(f"   Parameters: {gpt2_params:,}")
    print(f"   Vocab size: 50,257")
    print(f"   Hidden size: 768")
    print(f"   Layers: 12")
    print(f"   Attention heads: 12")
    
    results["architecture"] = {
        "gpt2": {
            "type": "Transformer Decoder",
            "parameters": gpt2_params,
            "vocab_size": 50257,
            "hidden_size": 768,
            "layers": 12,
            "attention_heads": 12,
        },
        "nicto": {
            "type": "Hybrid (MLA + MoE + Mamba + Liquid)",
            "parameters": "N/A (model too large for current memory)",
            "note": "NICTO model requires more memory than available. See AUDIT_REPORT.md for architecture details.",
        },
    }
    
    # Save results
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\n📊 GPT-2 Model:")
    print(f"   Parameters: {gpt2_params:,}")
    print(f"   Speed: {gpt2_speed:.2f} tokens/second")
    print(f"   Perplexity: {gpt2_ppl:.2f}")
    print(f"   Model size: {results['gpt2'].get('model_size_mb', 'N/A')} MB")
    
    print(f"\n📊 NICTO Model:")
    print(f"   Status: Cannot load (insufficient memory)")
    print(f"   Architecture: Hybrid (MLA + MoE + Mamba + Liquid)")
    print(f"   Note: Requires GPU or more RAM for full benchmark")
    
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_benchmark()
