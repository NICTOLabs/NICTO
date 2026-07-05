"""
NICTO AI - Data Pipeline
Tokenization, dataset loading, and data preprocessing for training
"""

import os
import json
import mmap
import struct
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class TokenizerWrapper:
    """
    Tokenizer wrapper that works with or without a real tokenizer.
    Falls back to simple byte-level encoding when tiktoken/huggingface unavailable.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._tokenizer = None
        self._backend = None
        self._try_load()

    def _try_load(self):
        """Try loading a real tokenizer."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            self._backend = "tiktoken"
            self.vocab_size = self._tokenizer.n_vocab
            return
        except Exception:
            pass

        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._backend = "huggingface"
            self.vocab_size = self._tokenizer.vocab_size
            return
        except Exception:
            pass

        self._backend = "fallback"

    def encode(self, text: str) -> List[int]:
        if self._backend == "tiktoken":
            return self._tokenizer.encode(text)
        elif self._backend == "huggingface":
            return self._tokenizer.encode(text, add_special_tokens=False)
        else:
            return self._fallback_encode(text)

    def decode(self, ids: List[int]) -> str:
        if self._backend == "tiktoken":
            return self._tokenizer.decode(ids)
        elif self._backend == "huggingface":
            return self._tokenizer.decode(ids)
        else:
            return self._fallback_decode(ids)

    def _fallback_encode(self, text: str) -> List[int]:
        """Byte-level encoding: map each byte to vocab slot."""
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def _fallback_decode(self, ids: List[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class PretokenizedDataset(Dataset):
    """
    Memory-mapped dataset for pretokenized .bin files.

    Expected format: each file is a flat array of uint16 or uint32 token IDs.
    """

    def __init__(self, data_path: str, seq_len: int = 2048):
        self.seq_len = seq_len
        self.data_path = Path(data_path)

        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            self.files = sorted(self.data_path.glob("*.bin"))

        if not self.files:
            raise FileNotFoundError(f"No .bin files found in {data_path}")

        # Load first file to determine dtype
        sample = torch.load(self.files[0], map_location="cpu")
        if sample.dtype == torch.uint16:
            self.dtype = torch.uint16
            self.bytes_per_token = 2
        else:
            self.dtype = torch.int32
            self.bytes_per_token = 4

        # Compute total length
        self.total_length = 0
        self.file_offsets = []
        for f in self.files:
            size = f.stat().st_size // self.bytes_per_token
            self.file_offsets.append(self.total_length)
            self.total_length += size

    def __len__(self):
        return max(0, (self.total_length - self.seq_len - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        tokens = []

        for i in range(self.seq_len + 1):
            pos = start + i
            # Find which file this token is in
            file_idx = 0
            for j, offset in enumerate(self.file_offsets):
                if j + 1 < len(self.file_offsets):
                    if pos < self.file_offsets[j + 1]:
                        file_idx = j
                        break
                else:
                    file_idx = j

            local_pos = pos - self.file_offsets[file_idx]
            with open(self.files[file_idx], "rb") as f:
                f.seek(local_pos * self.bytes_per_token)
                data = f.read(self.bytes_per_token)
                if self.dtype == torch.uint16:
                    token = struct.unpack("<H", data)[0]
                else:
                    token = struct.unpack("<i", data)[0]
            tokens.append(token)

        tokens = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class TextFileDataset(Dataset):
    """
    Dataset from raw text files. Tokenizes on-the-fly.
    """

    def __init__(self, data_path: str, seq_len: int = 2048, vocab_size: int = 32000):
        self.seq_len = seq_len
        self.tokenizer = TokenizerWrapper(vocab_size)

        self.data_path = Path(data_path)
        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            extensions = ["*.txt", "*.md", "*.jsonl", "*.json", "*.py", "*.ts"]
            self.files = []
            for ext in extensions:
                self.files.extend(self.data_path.glob(f"**/{ext}"))
            self.files = sorted(self.files)

        if not self.files:
            raise FileNotFoundError(f"No text files found in {data_path}")

        # Tokenize all files and concatenate
        all_tokens = []
        for f in self.files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for very large corpora.
    Reads data in chunks without loading everything into memory.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 2048,
        vocab_size: int = 32000,
        shuffle: bool = True,
    ):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.tokenizer = TokenizerWrapper(vocab_size)

        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            self.files = sorted(self.data_path.glob("*.txt"))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.files
        else:
            files = self.files[worker_info.id :: worker_info.num_workers]

        for filepath in files:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            tokens = self.tokenizer.encode(text)

            for i in range(0, len(tokens) - self.seq_len - 1, self.seq_len):
                chunk = torch.tensor(tokens[i : i + self.seq_len + 1], dtype=torch.long)
                yield {
                    "input_ids": chunk[:-1],
                    "labels": chunk[1:],
                }


def create_dataloader(
    data_path: str,
    seq_len: int = 2048,
    batch_size: int = 4,
    vocab_size: int = 32000,
    num_workers: int = 0,
    streaming: bool = False,
    distributed: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from a data path.

    Auto-detects format:
    - .bin files -> PretokenizedDataset
    - .txt/.md/.jsonl files -> TextFileDataset
    - Directories -> StreamingDataset

    Args:
        data_path: Path to data file or directory
        seq_len: Sequence length
        batch_size: Batch size
        vocab_size: Vocabulary size
        num_workers: Number of data loading workers
        streaming: Force streaming mode
        distributed: Use DistributedSampler

    Returns:
        DataLoader instance
    """
    path = Path(data_path)

    if streaming or (path.is_dir() and not list(path.glob("*.bin"))):
        dataset = StreamingDataset(data_path, seq_len, vocab_size)
        sampler = None
    elif list(path.glob("*.bin")):
        dataset = PretokenizedDataset(data_path, seq_len)
        sampler = None
    else:
        dataset = TextFileDataset(data_path, seq_len, vocab_size)
        sampler = None

    if distributed and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and not streaming),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )


def create_synthetic_dataset(
    n_samples: int = 1000,
    seq_len: int = 256,
    vocab_size: int = 32000,
) -> torch.utils.data.TensorDataset:
    """Create synthetic data for testing."""
    data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    return torch.utils.data.TensorDataset(input_ids, labels)
