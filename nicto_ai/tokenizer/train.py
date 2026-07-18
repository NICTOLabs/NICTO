"""
NICTO BPE Tokenizer Training
=============================
Trains a Byte-Pair Encoding tokenizer from scratch using the `tokenizers` library.
Uses the same architecture as LLaMA's tokenizer (ByteLevel BPE with prefix space).

Usage:
    # Train on text file(s)
    python -m nicto_ai.tokenizer.train --input training_data.jsonl --vocab-size 32000

    # Train on directory of text files
    python -m nicto_ai.tokenizer.train --input nicto_ai/data/processed/ --vocab-size 32000
"""

import os
import json
import argparse
from pathlib import Path
from typing import Iterator, Optional, Union

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def _iter_texts(data_path: Union[str, Path]) -> Iterator[str]:
    """
    Yield text strings from a file or directory.
    Supports: .txt, .md, .jsonl, .json files.
    For JSONL, reads the 'text' field.
    """
    path = Path(data_path)
    if path.is_file():
        ext = path.suffix.lower()
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if text and len(text) > 50:
                            yield text
                    except json.JSONDecodeError:
                        continue
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", "") if isinstance(item, dict) else str(item)
                        if text and len(text) > 50:
                            yield text
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                yield f.read()
    elif path.is_dir():
        for ext in ["*.txt", "*.md", "*.jsonl"]:
            for fp in path.rglob(ext):
                if fp.suffix == ".jsonl":
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                text = record.get("text", "")
                                if text and len(text) > 50:
                                    yield text
                            except json.JSONDecodeError:
                                continue
                else:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    if len(text) > 50:
                        yield text


def train_tokenizer(
    data_path: Union[str, Path],
    vocab_size: int = 32000,
    output_dir: Union[str, Path] = "nicto_ai/tokenizer/artifacts",
    min_frequency: int = 2,
    special_tokens: Optional[list] = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer on text data.

    This follows the same design as LLaMA's tokenizer:
    - ByteLevel pre-tokenization (splits on whitespace, punctuation)
    - BPE merges down to vocab_size tokens
    - ByteLevel decoder for clean output

    Args:
        data_path: Path to text file or directory of files
        vocab_size: Target vocabulary size (32000 = LLaMA standard)
        output_dir: Where to save tokenizer artifacts
        min_frequency: Minimum token frequency to include
        special_tokens: Additional special tokens beyond the defaults

    Returns:
        Trained Tokenizer instance
    """
    if special_tokens is None:
        special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    # Collect texts
    texts = list(_iter_texts(data_path))
    print(f"Training tokenizer on {len(texts)} documents from {data_path}")

    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    print(f"\nTokenizer trained successfully!")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  BOS token: <s> (id={bos_id})")
    print(f"  EOS token: </s> (id={eos_id})")
    print(f"  Saved to: {output_dir / 'tokenizer.json'}")

    # Quick test
    test_text = "Hello, world! 2 + 2 = 4"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n  Test encode: '{test_text}'")
    print(f"  Token IDs: {encoded.ids}")
    print(f"  Tokens: {encoded.tokens}")
    print(f"  Decoded: '{decoded}'")

    return tokenizer


def load_tokenizer(
    tokenizer_path: Union[str, Path] = "nicto_ai/tokenizer/artifacts/tokenizer.json",
) -> Tokenizer:
    """
    Load a trained tokenizer from disk.

    Args:
        tokenizer_path: Path to tokenizer.json

    Returns:
        Tokenizer instance
    """
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train NICTO BPE tokenizer")
    parser.add_argument("--input", "-i", required=True, help="Input text file or directory")
    parser.add_argument("--vocab-size", "-v", type=int, default=32000, help="Vocab size (default: 32000)")
    parser.add_argument("--output", "-o", default="nicto_ai/tokenizer/artifacts", help="Output directory")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency")
    args = parser.parse_args()

    train_tokenizer(
        data_path=args.input,
        vocab_size=args.vocab_size,
        output_dir=args.output,
        min_frequency=args.min_freq,
    )


if __name__ == "__main__":
    main()
