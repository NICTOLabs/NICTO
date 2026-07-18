"""Quick validation of downloaded training data."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from nicto_ai.training.data_pipeline_v2 import MMapDataset, MixedMMapDataset, create_dataloader

for name in ["oasst1", "dolly", "c4", "alpaca"]:
    path = f"training_data/{name}.bin"
    ds = MMapDataset(path, seq_len=2048)
    sample = ds[0]
    print(f"{name:8s}: {ds.n_tokens:>8,} tokens, {ds.n_samples:>5,} samples, "
          f"input_ids={list(sample['input_ids'].shape)}, labels={list(sample['labels'].shape)}")
    assert sample["input_ids"].shape == (2048,)
    assert sample["labels"].shape == (2048,)

mixed = MixedMMapDataset([
    ("training_data/oasst1.bin", 1.0),
    ("training_data/dolly.bin", 0.5),
    ("training_data/c4.bin", 3.0),
    ("training_data/alpaca.bin", 0.5),
], seq_len=2048)
print(f"\nMixed: {mixed.total_size:,} effective samples")

loader = create_dataloader([
    ("training_data/oasst1.bin", 1.0),
    ("training_data/dolly.bin", 0.5),
    ("training_data/c4.bin", 3.0),
    ("training_data/alpaca.bin", 0.5),
], seq_len=2048, batch_size=2)
batch = next(iter(loader))
print(f"Batch: input_ids={list(batch['input_ids'].shape)}, labels={list(batch['labels'].shape)}")
print("\n=== Data pipeline: ALL PASS ===")
