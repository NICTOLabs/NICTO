"""
NICTO Data Pipeline v2 — Memory-Mapped Pre-Tokenized Data
==========================================================
Industry-standard approach used by Megatron-LM, GPT-NeoX, LLaMA.

Pipeline:
  1. Pre-tokenize text → flat array of uint16 token IDs → save as .bin
  2. Memory-map .bin files for O(1) random access
  3. DataLoader slices contiguous chunks for training

This replaces the slow per-token file reads in the original pipeline.
"""

import json
import struct
import mmap
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _iter_texts(data_path: Union[str, Path]) -> Iterator[str]:
    """Yield text strings from a file or directory."""
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
            for fp in sorted(path.rglob(ext)):
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


def pretokenize(
    data_path: Union[str, Path],
    tokenizer_path: str,
    output_dir: Union[str, Path],
    chunk_size: int = 100_000,
    eos_token_id: int = 2,
) -> Path:
    """
    Pre-tokenize text data and save as .bin file(s).

    Args:
        data_path: Path to text file or directory
        tokenizer_path: Path to tokenizer.json
        output_dir: Where to save .bin files
        chunk_size: How many documents to process before flushing
        eos_token_id: Token ID for end-of-sequence

    Returns:
        Path to the .bin file
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens: List[int] = []
    doc_count = 0

    for text in _iter_texts(data_path):
        encoded = tokenizer.encode(text)
        all_tokens.extend(encoded.ids)
        all_tokens.append(eos_token_id)  # separator between documents
        doc_count += 1

        if doc_count % chunk_size == 0:
            print(f"  Tokenized {doc_count:,} documents, {len(all_tokens):,} tokens")

    # Save as uint16 (max vocab 65535, sufficient for 32K vocab)
    arr = np.array(all_tokens, dtype=np.uint16)
    bin_path = output_dir / "data.bin"
    arr.tofile(str(bin_path))

    print(f"\nPre-tokenization complete:")
    print(f"  Documents: {doc_count:,}")
    print(f"  Tokens: {len(all_tokens):,}")
    print(f"  File: {bin_path} ({bin_path.stat().st_size / 1e6:.1f} MB)")

    return bin_path


class MMapDataset(Dataset):
    """
    Memory-mapped dataset for pre-tokenized .bin files.

    Reads token IDs directly from memory-mapped numpy arrays.
    This is O(1) per sample — no file I/O during training.
    """

    def __init__(self, bin_path: Union[str, Path], seq_len: int = 2048):
        self.seq_len = seq_len
        self.bin_path = Path(bin_path)

        # Memory-map the file
        self.data = np.memmap(str(self.bin_path), dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        self.n_samples = max(0, (self.n_tokens - 1) // seq_len)

        print(f"MMapDataset: {self.n_tokens:,} tokens, {self.n_samples:,} samples, seq_len={seq_len}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].copy().astype(np.int64))
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class MixedMMapDataset(Dataset):
    """
    Mix multiple .bin files with weighted sampling.
    Each .bin file has its own weight.
    """

    def __init__(self, configs: List[tuple], seq_len: int = 2048):
        """
        Args:
            configs: List of (bin_path, weight) tuples
            seq_len: Sequence length
        """
        self.seq_len = seq_len
        self.datasets = []
        self.weights = []
        self.cumulative_sizes = []

        total_weight = sum(w for _, w in configs)

        for path, weight in configs:
            ds = MMapDataset(path, seq_len)
            self.datasets.append(ds)
            self.weights.append(weight / total_weight)

        # Build cumulative sizes for weighted sampling
        cumulative = 0
        for ds, w in zip(self.datasets, self.weights):
            cumulative += len(ds) * w
            self.cumulative_sizes.append(cumulative)

        self.total_size = int(cumulative)
        print(f"MixedMMapDataset: {len(self.datasets)} datasets, {self.total_size:,} effective samples")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        import random
        r = random.random() * self.total_size
        for i, cum_size in enumerate(self.cumulative_sizes):
            if r < cum_size:
                ds_idx = i
                break
        else:
            ds_idx = len(self.datasets) - 1

        sample_idx = random.randint(0, len(self.datasets[ds_idx]) - 1)
        return self.datasets[ds_idx][sample_idx]


def create_dataloader(
    data_source,
    seq_len: int = 2048,
    batch_size: int = 4,
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from a pre-tokenized .bin file or mixed configs.

    Args:
        data_source: Either a .bin path, or a list of (bin_path, weight) tuples
        seq_len: Sequence length
        batch_size: Batch size
        num_workers: Data loading workers (0 = main process)
        distributed: Use DistributedSampler for multi-GPU

    Returns:
        DataLoader instance
    """
    if isinstance(data_source, (str, Path)):
        dataset = MMapDataset(data_source, seq_len)
    elif isinstance(data_source, list):
        dataset = MixedMMapDataset(data_source, seq_len)
    else:
        raise ValueError(f"Unsupported data_source type: {type(data_source)}")

    sampler = None
    if distributed and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
