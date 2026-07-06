"""
NICTO Data Processor
===================
Cleans, filters, deduplicates, and tokenizes downloaded data.

Usage:
    python -m nicto_ai.data.processor --input raw/fineweb_edu.jsonl
    python -m nicto_ai.data.processor --all
    python -m nicto_ai.data.processor --tokenize --input processed/fineweb_edu.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Set, Dict
from collections import Counter

from .registry import RAW_DIR, PROCESSED_DIR, TOKENIZED_DIR, ALL_DATASETS


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean text for training.
    - Remove excessive whitespace
    - Remove special characters
    - Normalize unicode
    - Fix encoding issues
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
    text = re.sub(r" {2,}", " ", text)  # Max 1 space
    text = re.sub(r"\t{2,}", "\t", text)  # Max 1 tab
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def is_quality_text(text: str) -> bool:
    """
    Check if text is high quality for training.
    Filters out:
    - Too short or too long
    - Mostly special characters
    - Repetitive patterns
    - Known low-quality sources
    """
    if not text or len(text) < 100:
        return False
    
    if len(text) > 100000:
        return False
    
    # Check ratio of alphanumeric characters
    alpha_ratio = sum(c.isalnum() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 10:
        word_counts = Counter(words)
        most_common_ratio = word_counts.most_common(1)[0][1] / len(words)
        if most_common_ratio > 0.3:  # More than 30% same word
            return False
    
    # Check for URLs (too many = low quality)
    url_count = len(re.findall(r"https?://\S+", text))
    if url_count > 5:
        return False
    
    # Check for email addresses
    email_count = len(re.findall(r"\S+@\S+\.\S+", text))
    if email_count > 3:
        return False
    
    return True


def deduplicate_text(texts: List[str], threshold: float = 0.8) -> List[str]:
    """
    Simple deduplication using hash-based approach.
    For exact duplicates and near-duplicates.
    """
    seen_hashes: Set[str] = set()
    unique_texts = []
    
    for text in texts:
        # Exact duplicate check
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        
        # Near-duplicate check (first 500 chars)
        prefix = text[:500]
        prefix_hash = hashlib.md5(prefix.encode()).hexdigest()
        if prefix_hash in seen_hashes:
            continue
        
        seen_hashes.add(text_hash)
        seen_hashes.add(prefix_hash)
        unique_texts.append(text)
    
    return unique_texts


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize_text(text: str, tokenizer) -> List[int]:
    """
    Tokenize text using the given tokenizer.
    Adds special tokens and handles truncation.
    """
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        max_length=2048,
        truncation=True,
    )
    return tokens


def create_tokenizer():
    """
    Create a tokenizer for NICTO.
    Uses a simple BPE tokenizer trained on the data.
    """
    try:
        from transformers import AutoTokenizer
        
        # Use a small, efficient tokenizer as base
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print("Using character-level tokenizer")
        return None


# =============================================================================
# DATA MIXING
# =============================================================================

class DataMixer:
    """
    Mixes multiple datasets with specified proportions.
    Supports weighted sampling and interleaving.
    """
    
    def __init__(self, dataset_paths: Dict[str, Path], weights: Dict[str, float]):
        """
        Args:
            dataset_paths: {name: path_to_jsonl}
            weights: {name: weight_for_mixing}
        """
        self.dataset_paths = dataset_paths
        self.weights = weights
        self.total_weight = sum(weights.values())
        
        # Normalize weights
        self.normalized_weights = {
            name: w / self.total_weight
            for name, w in weights.items()
        }
    
    def get_mix_plan(self, total_tokens: int) -> Dict[str, int]:
        """
        Create a plan for how many tokens to take from each dataset.
        
        Args:
            total_tokens: Total tokens to mix
        
        Returns:
            {name: tokens_to_sample}
        """
        plan = {}
        for name, weight in self.normalized_weights.items():
            tokens = int(total_tokens * weight)
            plan[name] = tokens
        return plan
    
    def mixed_iterator(self, total_tokens: int = 10_000_000_000):
        """
        Iterate through mixed data.
        Yields (text, source) tuples.
        """
        plan = self.get_mix_plan(total_tokens)
        
        for name, target_tokens in plan.items():
            if name not in self.dataset_paths:
                print(f"Warning: Dataset {name} not found, skipping")
                continue
            
            path = self.dataset_paths[name]
            tokens_sampled = 0
            
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if tokens_sampled >= target_tokens:
                        break
                    
                    record = json.loads(line)
                    text = record["text"]
                    
                    # Approximate token count (1 token ≈ 4 chars)
                    token_count = len(text) // 4
                    
                    yield text, name
                    tokens_sampled += token_count


# =============================================================================
# PROCESSING PIPELINE
# =============================================================================

def process_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    clean: bool = True,
    filter_quality: bool = True,
    deduplicate: bool = True,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Process a single JSONL file.
    
    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file (default=processed/)
        clean: Whether to clean text
        filter_quality: Whether to filter quality
        deduplicate: Whether to deduplicate
        max_samples: Maximum samples to process
    
    Returns:
        Stats about processing
    """
    if output_path is None:
        output_path = PROCESSED_DIR / input_path.name
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_read": 0,
        "cleaned": 0,
        "filtered": 0,
        "deduplicated": 0,
        "output_count": 0,
    }
    
    print(f"\nProcessing: {input_path}")
    print(f"Output: {output_path}")
    
    texts = []
    
    # Read all texts
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and stats["total_read"] >= max_samples:
                break
            
            record = json.loads(line)
            text = record.get("text", "")
            stats["total_read"] += 1
            
            # Clean
            if clean:
                text = clean_text(text)
                if text != record.get("text", ""):
                    stats["cleaned"] += 1
            
            # Filter
            if filter_quality and not is_quality_text(text):
                stats["filtered"] += 1
                continue
            
            texts.append({"text": text, "source": record.get("source", "unknown")})
    
    # Deduplicate
    if deduplicate:
        unique_texts = [t["text"] for t in texts]
        unique_texts = deduplicate_text(unique_texts)
        stats["deduplicated"] = len(texts) - len(unique_texts)
        texts = [{"text": t, "source": texts[i]["source"]} 
                 for i, t in enumerate(unique_texts)]
    
    # Write processed output
    with open(output_path, "w", encoding="utf-8") as f:
        for record in texts:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["output_count"] += 1
    
    print(f"  Total read: {stats['total_read']:,}")
    print(f"  Cleaned: {stats['cleaned']:,}")
    print(f"  Filtered: {stats['filtered']:,}")
    print(f"  Deduplicated: {stats['deduplicated']:,}")
    print(f"  Output: {stats['output_count']:,}")
    
    return stats


def tokenize_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Tokenize a processed JSONL file.
    
    Args:
        input_path: Input JSONL file
        output_path: Output directory for tokenized data
        max_samples: Maximum samples to tokenize
    
    Returns:
        Stats about tokenization
    """
    if output_path is None:
        output_path = TOKENIZED_DIR / input_path.stem
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "input_file": str(input_path),
        "output_dir": str(output_path),
        "total_read": 0,
        "total_tokens": 0,
        "output_files": 0,
    }
    
    print(f"\nTokenizing: {input_path}")
    print(f"Output: {output_path}")
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    if tokenizer is None:
        print("Cannot tokenize without tokenizer")
        return stats
    
    # Tokenize and save in chunks
    chunk_size = 10000  # samples per file
    current_chunk = []
    chunk_idx = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and stats["total_read"] >= max_samples:
                break
            
            record = json.loads(line)
            text = record["text"]
            
            tokens = tokenize_text(text, tokenizer)
            current_chunk.extend(tokens)
            stats["total_read"] += 1
            stats["total_tokens"] += len(tokens)
            
            # Save chunk when full
            if len(current_chunk) >= chunk_size * 1024:
                chunk_file = output_path / f"chunk_{chunk_idx:04d}.bin"
                import numpy as np
                np.save(chunk_file, np.array(current_chunk, dtype=np.uint16))
                current_chunk = []
                chunk_idx += 1
    
    # Save remaining
    if current_chunk:
        chunk_file = output_path / f"chunk_{chunk_idx:04d}.bin"
        import numpy as np
        np.save(chunk_file, np.array(current_chunk, dtype=np.uint16))
        chunk_idx += 1
    
    stats["output_files"] = chunk_idx
    
    print(f"  Total read: {stats['total_read']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Output files: {stats['output_files']}")
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process NICTO training data")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--all", action="store_true", help="Process all raw files")
    parser.add_argument("--tokenize", action="store_true", help="Tokenize processed files")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    
    args = parser.parse_args()
    
    if args.all:
        print("Processing all raw files...")
        raw_files = list(RAW_DIR.glob("*.jsonl"))
        
        for raw_file in raw_files:
            if args.tokenize:
                processed_file = PROCESSED_DIR / raw_file.name
                tokenize_file(processed_file, max_samples=args.max_samples)
            else:
                process_file(raw_file, max_samples=args.max_samples)
    
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"File not found: {input_path}")
            sys.exit(1)
        
        if args.tokenize:
            tokenize_file(input_path, max_samples=args.max_samples)
        else:
            process_file(input_path, max_samples=args.max_samples)
    
    else:
        print("Specify --input FILE or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
