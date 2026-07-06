"""
NICTO Data Downloader
====================
Downloads datasets from HuggingFace Hub.
Saves raw data as JSONL files for processing.

Usage:
    python -m nicto_ai.data.downloader --dataset "FineWeb-Edu"
    python -m nicto_ai.data.downloader --all
    python -m nicto_ai.data.downloader --priority 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List
from datasets import load_dataset, Dataset
from tqdm import tqdm

from .registry import (
    RAW_DIR,
    DatasetConfig,
    ALL_DATASETS,
    PRIORITY_1,
    get_dataset_by_name,
)


def download_dataset(
    config: DatasetConfig,
    max_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Download a dataset from HuggingFace and save as JSONL.
    
    Args:
        config: Dataset configuration
        max_samples: Maximum number of samples to download (None=all)
        output_dir: Output directory (default=RAW_DIR)
    
    Returns:
        Path to saved JSONL file
    """
    if output_dir is None:
        output_dir = RAW_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = config.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_file = output_dir / f"{safe_name}.jsonl"
    
    print(f"\n{'=' * 60}")
    print(f"Downloading: {config.name}")
    print(f"HuggingFace ID: {config.hf_id}")
    print(f"Split: {config.split}")
    print(f"Output: {output_file}")
    print(f"{'=' * 60}")
    
    try:
        # Load dataset
        print("Loading dataset from HuggingFace...")
        kwargs = {"split": config.split}
        if config.subset:
            kwargs["name"] = config.subset
        
        dataset = load_dataset(config.hf_id, split=kwargs["split"], streaming=True)
        
        # Get total size if possible
        try:
            total = dataset.info.splits[config.split].num_examples
        except (AttributeError, KeyError):
            total = max_samples or "unknown"
        
        print(f"Total examples: {total}")
        
        # Iterate and save
        count = 0
        skipped = 0
        
        with open(output_file, "w", encoding="utf-8") as f:
            for example in tqdm(dataset, total=max_samples or total, desc=config.name):
                # Extract text field
                if config.text_field not in example:
                    skipped += 1
                    continue
                
                text = example[config.text_field]
                
                # Handle conversations (list of dicts)
                if isinstance(text, list):
                    # Convert conversation to text
                    parts = []
                    for turn in text:
                        if isinstance(turn, dict):
                            role = turn.get("role", "user")
                            content = turn.get("content", "")
                            parts.append(f"{role}: {content}")
                    text = "\n".join(parts)
                
                if not isinstance(text, str):
                    skipped += 1
                    continue
                
                # Filter by length
                if len(text) < config.min_length:
                    skipped += 1
                    continue
                if len(text) > config.max_length:
                    skipped += 1
                    continue
                
                # Filter by field value
                if config.filter_field and config.filter_value:
                    if example.get(config.filter_field) != config.filter_value:
                        skipped += 1
                        continue
                
                # Write JSONL
                record = {"text": text, "source": config.name}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                count += 1
                if max_samples and count >= max_samples:
                    break
        
        print(f"\nSaved {count:,} examples to {output_file}")
        print(f"Skipped {skipped:,} examples")
        
        return output_file
    
    except Exception as e:
        print(f"Error downloading {config.name}: {e}")
        raise


def download_all(
    datasets: Optional[List[DatasetConfig]] = None,
    max_samples_per_dataset: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Download multiple datasets.
    
    Args:
        datasets: List of dataset configs (default=all)
        max_samples_per_dataset: Max samples per dataset
        output_dir: Output directory
    
    Returns:
        List of paths to saved files
    """
    if datasets is None:
        datasets = ALL_DATASETS
    
    paths = []
    for config in datasets:
        try:
            path = download_dataset(
                config,
                max_samples=max_samples_per_dataset,
                output_dir=output_dir,
            )
            paths.append(path)
        except Exception as e:
            print(f"Failed to download {config.name}: {e}")
            continue
    
    return paths


def main():
    parser = argparse.ArgumentParser(description="Download NICTO training data")
    parser.add_argument("--dataset", type=str, help="Dataset name to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--priority", type=int, help="Download all datasets of priority N")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else RAW_DIR
    
    if args.all:
        print("Downloading ALL datasets...")
        paths = download_all(max_samples_per_dataset=args.max_samples, output_dir=output_dir)
        print(f"\nDownloaded {len(paths)} datasets")
    
    elif args.priority:
        print(f"Downloading Priority {args.priority} datasets...")
        if args.priority == 1:
            datasets = PRIORITY_1
        elif args.priority == 2:
            from .registry import PRIORITY_2
            datasets = PRIORITY_2
        elif args.priority == 3:
            from .registry import PRIORITY_3
            datasets = PRIORITY_3
        else:
            from .registry import PRIORITY_4
            datasets = PRIORITY_4
        
        paths = download_all(datasets=datasets, max_samples_per_dataset=args.max_samples, output_dir=output_dir)
        print(f"\nDownloaded {len(paths)} datasets")
    
    elif args.dataset:
        config = get_dataset_by_name(args.dataset)
        if config is None:
            print(f"Unknown dataset: {args.dataset}")
            print("Available datasets:")
            for d in ALL_DATASETS:
                print(f"  - {d.name}")
            sys.exit(1)
        
        download_dataset(config, max_samples=args.max_samples, output_dir=output_dir)
    
    else:
        print("Specify --dataset NAME, --priority N, or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
