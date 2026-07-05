"""
NICTO AI vs GPT-2 Benchmark Comparison
Compare model size, speed, memory, and generation quality
"""

import torch
import time
import sys
import os
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nicto_ai.core.model import NICTOModel


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


class NICTOvsGPT2Benchmark:
    """
    Benchmark comparison between NICTO AI and GPT-2
    
    Tests:
    1. Parameter count
    2. Inference speed (tokens/second)
    3. Memory usage
    4. Text generation quality
    5. Training efficiency (theoretical)
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results: List[BenchmarkResult] = []
        
    def create_nicto_small(self) -> NICTOModel:
        """Create small NICTO model for comparison"""
        print("Creating small NICTO model...")
        model = NICTOModel(
            vocab_size=1000,
            dim=256,
            max_seq_len=512,
        )
        return model
    
    def load_gpt2(self):
        """Load GPT-2 model"""
        print("Loading GPT-2 model...")
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.to(self.device)
            model.eval()
            return model, tokenizer
        except ImportError:
            print("transformers not installed, using mock GPT-2")
            return None, None
    
    def count_parameters(self, model) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in model.parameters())
    
    def count_trainable_parameters(self, model) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def measure_inference_speed(self, model, input_ids, n_runs: int = 10) -> float:
        """Measure inference speed in tokens/second"""
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(input_ids)
        end_time = time.time()
        
        total_time = end_time - start_time
        tokens_per_second = (input_ids.shape[1] * n_runs) / total_time
        return tokens_per_second
    
    def measure_memory_usage(self, model, input_ids) -> Dict[str, float]:
        """Measure memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Before inference
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference
        with torch.no_grad():
            _ = model(input_ids)
        
        # After inference
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
        
        return {
            "before_mb": mem_before,
            "after_mb": mem_after,
            "delta_mb": mem_after - mem_before,
            "model_size_mb": model_size,
        }
    
    def generate_text(self, model, tokenizer, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt"""
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if hasattr(model, 'generate'):
                # GPT-2 style
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
            else:
                # NICTO style
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_length - input_ids.shape[1],
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    def calculate_perplexity(self, model, tokenizer, text: str) -> float:
        """Calculate perplexity on text"""
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        
        with torch.no_grad():
            if hasattr(model, 'forward') and 'labels' in model.forward.__code__.co_varnames:
                # GPT-2 style
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            else:
                # NICTO style
                outputs = model(input_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        
        return torch.exp(loss).item()
    
    def run_benchmark(self) -> Dict:
        """Run complete benchmark"""
        print("=" * 60)
        print("NICTO AI vs GPT-2 Benchmark")
        print("=" * 60)
        
        # Create models
        nicto_model = self.create_nicto_small()
        gpt2_model, gpt2_tokenizer = self.load_gpt2()
        
        # Create tokenizer for NICTO (simple fallback)
        class SimpleTokenizer:
            def __init__(self, vocab_size=32000):
                self.vocab_size = vocab_size
            def encode(self, text, return_tensors=None):
                tokens = [ord(c) % self.vocab_size for c in text]
                if return_tensors == "pt":
                    return torch.tensor([tokens], dtype=torch.long)
                return tokens
            def decode(self, tokens, skip_special_tokens=True):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens[0].tolist()
                return "".join(chr(t % 128) for t in tokens)
        
        nicto_tokenizer = SimpleTokenizer()
        
        # Test prompts
        test_prompts = [
            "The quick brown fox",
            "Artificial intelligence is",
            "In the future, technology will",
            "The meaning of life is",
            "Python programming is",
        ]
        
        # Run tests
        results = {}
        
        # 1. Parameter count
        print("\n1. Parameter Count...")
        nicto_params = self.count_parameters(nicto_model)
        gpt2_params = self.count_parameters(gpt2_model) if gpt2_model else 0
        
        results["parameters"] = {
            "nicto": nicto_params,
            "gpt2": gpt2_params,
            "ratio": nicto_params / gpt2_params if gpt2_params > 0 else 0,
        }
        print(f"   NICTO: {nicto_params:,} parameters")
        print(f"   GPT-2: {gpt2_params:,} parameters")
        print(f"   Ratio: {results['parameters']['ratio']:.2f}x")
        
        # 2. Inference speed
        print("\n2. Inference Speed...")
        input_ids = torch.randint(0, 32000, (1, 128)).to(self.device)
        
        nicto_speed = self.measure_inference_speed(nicto_model, input_ids)
        gpt2_speed = self.measure_inference_speed(gpt2_model, input_ids) if gpt2_model else 0
        
        results["speed"] = {
            "nicto_tps": nicto_speed,
            "gpt2_tps": gpt2_speed,
            "ratio": nicto_speed / gpt2_speed if gpt2_speed > 0 else 0,
        }
        print(f"   NICTO: {nicto_speed:.2f} tokens/second")
        print(f"   GPT-2: {gpt2_speed:.2f} tokens/second")
        print(f"   Ratio: {results['speed']['ratio']:.2f}x")
        
        # 3. Memory usage
        print("\n3. Memory Usage...")
        try:
            nicto_mem = self.measure_memory_usage(nicto_model, input_ids)
            gpt2_mem = self.measure_memory_usage(gpt2_model, input_ids) if gpt2_model else {"model_size_mb": 0}
            
            results["memory"] = {
                "nicto_model_mb": nicto_mem["model_size_mb"],
                "gpt2_model_mb": gpt2_mem["model_size_mb"],
                "ratio": nicto_mem["model_size_mb"] / gpt2_mem["model_size_mb"] if gpt2_mem["model_size_mb"] > 0 else 0,
            }
            print(f"   NICTO: {nicto_mem['model_size_mb']:.2f} MB")
            print(f"   GPT-2: {gpt2_mem['model_size_mb']:.2f} MB")
            print(f"   Ratio: {results['memory']['ratio']:.2f}x")
        except ImportError:
            print("   psutil not installed, skipping memory test")
            results["memory"] = {"nicto_model_mb": 0, "gpt2_model_mb": 0, "ratio": 0}
        
        # 4. Text generation
        print("\n4. Text Generation...")
        generation_results = {}
        
        for prompt in test_prompts[:2]:  # Test with first 2 prompts
            print(f"\n   Prompt: '{prompt}'")
            
            # NICTO generation
            nicto_output = self.generate_text(nicto_model, nicto_tokenizer, prompt, max_length=100)
            print(f"   NICTO: {nicto_output[:100]}...")
            
            # GPT-2 generation
            if gpt2_model and gpt2_tokenizer:
                gpt2_output = self.generate_text(gpt2_model, gpt2_tokenizer, prompt, max_length=100)
                print(f"   GPT-2: {gpt2_output[:100]}...")
            else:
                gpt2_output = "N/A"
            
            generation_results[prompt] = {
                "nicto": nicto_output,
                "gpt2": gpt2_output,
            }
        
        results["generation"] = generation_results
        
        # 5. Perplexity
        print("\n5. Perplexity...")
        test_text = "Artificial intelligence is transforming the way we live and work. Machine learning algorithms can now recognize images, understand natural language, and make decisions."
        
        nicto_ppl = self.calculate_perplexity(nicto_model, nicto_tokenizer, test_text)
        gpt2_ppl = self.calculate_perplexity(gpt2_model, gpt2_tokenizer, test_text) if gpt2_model and gpt2_tokenizer else 0
        
        results["perplexity"] = {
            "nicto": nicto_ppl,
            "gpt2": gpt2_ppl,
            "lower_is_better": True,
        }
        print(f"   NICTO: {nicto_ppl:.2f}")
        print(f"   GPT-2: {gpt2_ppl:.2f}")
        
        # 6. Architecture comparison
        print("\n6. Architecture Comparison...")
        results["architecture"] = {
            "nicto": {
                "type": "Hybrid (MLA + MoE + Mamba + Liquid)",
                "attention": "Multi-Latent Attention (MLA)",
                "ffn": "Mixture of Experts (MoE)",
                "sequence_modeling": "Mamba (State Space Model)",
                "special": "Liquid Neural Networks, Consciousness Layer",
                "parameters": f"{nicto_params:,}",
            },
            "gpt2": {
                "type": "Transformer Decoder",
                "attention": "Multi-Head Attention",
                "ffn": "Standard FFN",
                "sequence_modeling": "Self-Attention",
                "special": "None",
                "parameters": f"{gpt2_params:,}",
            },
        }
        
        # Save results
        output_file = Path(__file__).parent / "benchmark_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        print("\n📊 Model Size:")
        print(f"   NICTO: {results['parameters']['nicto']:,} parameters")
        print(f"   GPT-2: {results['parameters']['gpt2']:,} parameters")
        print(f"   NICTO is {results['parameters']['ratio']:.2f}x larger")
        
        print("\n⚡ Inference Speed:")
        print(f"   NICTO: {results['speed']['nicto_tps']:.2f} tokens/second")
        print(f"   GPT-2: {results['speed']['gpt2_tps']:.2f} tokens/second")
        if results['speed']['ratio'] > 0:
            print(f"   NICTO is {results['speed']['ratio']:.2f}x faster")
        
        print("\n💾 Memory Usage:")
        print(f"   NICTO: {results['memory']['nicto_model_mb']:.2f} MB")
        print(f"   GPT-2: {results['memory']['gpt2_model_mb']:.2f} MB")
        if results['memory']['ratio'] > 0:
            print(f"   NICTO uses {results['memory']['ratio']:.2f}x more memory")
        
        print("\n📈 Perplexity (Lower is Better):")
        print(f"   NICTO: {results['perplexity']['nicto']:.2f}")
        print(f"   GPT-2: {results['perplexity']['gpt2']:.2f}")
        
        print("\n🏗️ Architecture:")
        print(f"   NICTO: {results['architecture']['nicto']['type']}")
        print(f"   GPT-2: {results['architecture']['gpt2']['type']}")
        
        print("\n" + "=" * 60)


def main():
    """Main function"""
    benchmark = NICTOvsGPT2Benchmark(device="cpu")
    results = benchmark.run_benchmark()
    benchmark.print_summary(results)
    
    print("\n✅ Benchmark complete!")
    print("Results saved to: tests/benchmark_results.json")


if __name__ == "__main__":
    main()
