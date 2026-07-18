"""
NICTO Data Preparation Pipeline
================================
Downloads, cleans, and pre-tokenizes open-source datasets for training.

Target datasets (all permissive licenses):
  1. OpenAssistant OASST1  — instruction conversations  (Apache-2.0)
  2. Dolly 15K              — human-written QA           (CC-BY-SA)
  3. FineWeb sample         — high-quality web text      (ODC-By)
  4. Wikipedia (en)         — encyclopedia               (CC-BY-SA)
  5. The Stack (code)       — permissive-licensed code   (various)

Usage:
  uv run python prepare_data.py --datasets oasst1,dolly,fineweb,wikipedia,code
  uv run python prepare_data.py --all          # download everything
  uv run python prepare_data.py --quick        # small test sample only
"""

import argparse, json, math, os, random, sys, time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 on Windows
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "training_data"
TOKENIZER_PATH = Path(__file__).parent / "nicto_ai" / "tokenizer" / "artifacts" / "tokenizer.json"
EOS_ID = 2

DATA_DIR.mkdir(exist_ok=True)


# ── Text Cleaning ──────────────────────────────────────────────

def clean_text(text: str) -> Optional[str]:
    """Clean and filter a text sample. Returns None if it should be skipped."""
    if not text or not isinstance(text, str):
        return None
    # Truncate BEFORE any string ops to prevent OOM on giant web pages
    if len(text) > 200_000:
        text = text[:200_000]
    text = text.strip()
    # Remove empty or tiny texts
    if len(text) < 50:
        return None
    # Remove texts that are just boilerplate
    boilerplate = [
        "javascript", "please enable javascript", "click here",
        "terms of service", "privacy policy", "all rights reserved",
    ]
    # Safety check before lower()
    if len(text) > 300_000:
        text = text[:200_000]
    text_lower = text.lower()
    if any(b in text_lower for b in boilerplate) and len(text) < 200:
        return None
    # Normalize whitespace
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text = "\n".join(lines)
    # Filter by character diversity (at least 5% unique — removes garbled data)
    if len(set(text)) < 0.05 * len(text):
        return None
    return text


def extract_text(record: dict, dataset: str) -> Optional[str]:
    """Extract the main text from a dataset record."""
    if dataset == "oasst1":
        # OASST1: include both user and assistant messages
        text = record.get("text", "")
        if text:
            return clean_text(text)
        return None
    elif dataset == "dolly":
        # Dolly: instruction + context + response (field names may vary)
        parts = []
        for key in ["instruction", "context", "response", "text", "prompt", "completion"]:
            val = record.get(key, "")
            if val and isinstance(val, str):
                parts.append(val)
        text = "\n".join(parts)
        return clean_text(text) if len(text) > 30 else None
    elif dataset == "fineweb":
        text = record.get("text", "")
        return clean_text(text)
    elif dataset == "wikipedia":
        title = record.get("title", "")
        text = record.get("text", "")
        if text:
            full = f"{title}\n\n{text}" if title else text
            return clean_text(full)
        return None
    elif dataset == "code":
        text = record.get("content", "") or record.get("text", "")
        return clean_text(text) if text else None
    elif dataset == "cosmopedia":
        text = record.get("text", "")
        return clean_text(text) if text else None
    elif dataset == "c4":
        text = record.get("text", "")
        return clean_text(text) if text else None
    elif dataset == "alpaca":
        parts = []
        for key in ["instruction", "input", "output"]:
            val = record.get(key, "")
            if val and isinstance(val, str):
                parts.append(val)
        text = "\n\n".join(parts)
        return clean_text(text) if text and len(text) > 30 else None
    elif dataset == "openorca":
        parts = []
        for key in ["question", "response", "system_prompt"]:
            val = record.get(key, "")
            if val and isinstance(val, str):
                parts.append(val)
        text = "\n\n".join(parts)
        return clean_text(text) if text and len(text) > 30 else None
    return None


# ── Dataset Loaders ─────────────────────────────────────────────

DATASET_CONFIGS = {
    "oasst1": {
        "hf_path": "OpenAssistant/oasst1",
        "split": "train",
        "description": "OpenAssistant conversations (Apache-2.0)",
        "weight": 1.0,
    },
    "dolly": {
        "hf_path": "databricks/databricks-dolly-15k",
        "split": "train",
        "description": "Dolly 15K human-written QA (CC-BY-SA)",
        "weight": 0.5,
    },
    "c4": {
        "hf_path": "allenai/c4",
        "split": "train",
        "description": "C4 - Colossal Clean Crawled Corpus (ODC-By)",
        "weight": 3.0,
        "subset": "en",
        "max_samples": 100_000,
    },
    "alpaca": {
        "hf_path": "tatsu-lab/alpaca",
        "split": "train",
        "description": "Alpaca 52K instructions (CC-BY-NC 4.0)",
        "weight": 0.5,
    },
    "openorca": {
        "hf_path": "Open-Orca/OpenOrca",
        "split": "train",
        "description": "OpenOrca instruction data (Apache-2.0)",
        "weight": 1.0,
        "max_samples": 200_000,
    },
}


class StreamingDataset:
    """Stream dataset from HuggingFace without downloading full corpus."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._iter = None

    def __iter__(self):
        return self._stream()

    def _stream(self):
        from datasets import load_dataset
        kwargs = {"path": self.config["hf_path"], "split": self.config["split"],
                   "streaming": True}
        if "subset" in self.config:
            kwargs["data_dir"] = self.config["subset"]
        ds = load_dataset(**kwargs)
        max_s = self.config.get("max_samples")
        for i, record in enumerate(ds):
            if max_s and i >= max_s:
                break
            text = extract_text(record, self.name)
            if text:
                yield text


# ── Pre-tokenization ────────────────────────────────────────────

def pretokenize_stream(text_stream, name: str,
                       max_tokens: int = 100_000_000) -> Path:
    """Tokenize texts from a stream and save to .bin file.

    Writes each document directly to the file to avoid OOM.
    """
    from tokenizers import Tokenizer

    print(f"\n  Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    bin_path = DATA_DIR / f"{name}.bin"
    meta_path = DATA_DIR / f"{name}_meta.json"

    if bin_path.exists():
        bin_path.unlink()

    import numpy as np

    doc_count = 0
    tok_count = 0
    t0 = time.time()
    f = open(bin_path, "wb")

    try:
        for text in text_stream:
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            if len(ids) > 50_000:
                ids = ids[:50_000]
            # Write directly: each doc as uint16 array
            arr = np.array(ids + [EOS_ID], dtype=np.uint16)
            arr.tofile(f)
            doc_count += 1
            tok_count += len(ids) + 1

            if doc_count % 10_000 == 0:
                elapsed = time.time() - t0
                tps = tok_count / max(elapsed, 0.1)
                print(f"    {doc_count:>8,d} docs | {tok_count:>10,d} tokens | {tps:>8,.0f} tok/s", end="\r")

            if tok_count >= max_tokens:
                break
    finally:
        f.close()

    tokens = bin_path.stat().st_size // 2
    print(f"\n    Done: {tokens:,} tokens in {bin_path.name}")

    meta = {
        "dataset": name,
        "documents": doc_count,
        "tokens": tokens,
        "file": str(bin_path),
        "size_mb": bin_path.stat().st_size / 1e6,
        "time_seconds": round(time.time() - t0, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return bin_path


# ── Dataset Mixing ──────────────────────────────────────────────

def create_mixed_dataset(bin_files: list[Path],
                         output_name: str = "nicto_mixed",
                         seq_len: int = 2048):
    """Create a merged dataset with weighted sampling config."""
    configs = []
    for bf in bin_files:
        meta_path = bf.with_name(bf.stem + "_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            weight = DATASET_CONFIGS.get(meta["dataset"], {}).get("weight", 1.0)
            configs.append((str(bf), weight))
            print(f"  {meta['dataset']:12s}  {meta['tokens']/1e6:>8.1f}M tokens  weight={weight}")

    config_path = DATA_DIR / f"{output_name}_config.json"
    with open(config_path, "w") as f:
        json.dump({"seq_len": seq_len, "sources": configs}, f, indent=2)
    print(f"\nConfig saved to {config_path}")
    return config_path


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NICTO Data Preparation Pipeline")
    parser.add_argument("--datasets", type=str, default="",
                        help="Comma-separated: oasst1,dolly,fineweb,wikipedia,code")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--quick", action="store_true", help="Download tiny test sample")
    parser.add_argument("--max-tokens", type=int, default=100_000_000,
                        help="Max tokens per dataset (default 100M)")
    args = parser.parse_args()

    if args.quick:
        selected = ["oasst1"]
        max_tokens = 1_000_000
        print("=== QUICK MODE: 1M tokens from OASST1 ===\n")
    elif args.all:
        selected = list(DATASET_CONFIGS.keys())
        max_tokens = args.max_tokens
        print(f"=== FULL DOWNLOAD: {', '.join(selected)} ===\n")
    else:
        selected = [s.strip() for s in args.datasets.split(",") if s.strip()]
        max_tokens = args.max_tokens
        print(f"=== SELECTED: {', '.join(selected)} ===\n")

    if not selected:
        print("No datasets selected. Use --all, --quick, or --datasets oasst1,dolly,...")
        return

    # Install dependencies if needed
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        os.system(f"{sys.executable} -m pip install datasets -q")
        from datasets import load_dataset

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("Installing 'tokenizers' library...")
        os.system(f"{sys.executable} -m pip install tokenizers -q")
        from tokenizers import Tokenizer

    bin_files = []
    for name in selected:
        if name not in DATASET_CONFIGS:
            print(f"\n  Unknown dataset: {name}. Skipping.")
            continue

        cfg = DATASET_CONFIGS[name]
        print(f"\n{'─'*60}")
        print(f"  [{name}] {cfg['description']}")
        print(f"  Source: {cfg['hf_path']} ({cfg['split']})")
        print(f"{'─'*60}")

        print(f"  Streaming and tokenizing (max {max_tokens/1e6:.0f}M tokens)...")
        stream = StreamingDataset(name, cfg)
        bin_path = pretokenize_stream(stream, name, max_tokens=max_tokens)
        bin_files.append(bin_path)

        tokens = bin_path.stat().st_size // 2  # uint16 = 2 bytes
        print(f"  ✓ Saved: {bin_path.name} ({tokens/1e6:.1f}M tokens, "
              f"{bin_path.stat().st_size/1e6:.1f} MB)")

    if len(bin_files) > 1:
        print(f"\n{'─'*60}")
        print("  Creating mixed dataset config...")
        create_mixed_dataset(bin_files)

    print(f"\n{'='*60}")
    print(f"  All data saved to {DATA_DIR}/")
    print(f"  Ready for training!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
