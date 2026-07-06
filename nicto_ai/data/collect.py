"""
NICTO Data Collection - Main Script
===================================
Downloads, processes, and prepares data for NICTO training.

Usage:
    # Download Priority 1 datasets
    python -m nicto_ai.data.collect --priority 1 --max-samples 100000
    
    # Download all datasets
    python -m nicto_ai.data.collect --all
    
    # Process downloaded data
    python -m nicto_ai.data.collect --process
    
    # Full pipeline (download + process)
    python -m nicto_ai.data.collect --full --priority 1
"""

import argparse
import sys
from pathlib import Path

from .registry import ALL_DATASETS, PRIORITY_1, PRIORITY_2, PRIORITY_3, PRIORITY_4
from .downloader import download_all
from .processor import process_file, tokenize_file
from .mix_config import T4_SMALL, T4_MEDIUM, T4_LARGE, K8S_FULL


def collect_data(
    priority: int = None,
    dataset_name: str = None,
    max_samples: int = None,
    process: bool = False,
    tokenize: bool = False,
    config_name: str = "t4_medium",
):
    """
    Main data collection pipeline.
    
    Args:
        priority: Download all datasets of this priority (1-4)
        dataset_name: Download specific dataset
        max_samples: Max samples per dataset
        process: Whether to process after downloading
        tokenize: Whether to tokenize after processing
        config_name: Mix config name (t4_small, t4_medium, t4_large, k8s_full)
    """
    # Select datasets
    if dataset_name:
        datasets = [d for d in ALL_DATASETS if d.name == dataset_name]
        if not datasets:
            print(f"Unknown dataset: {dataset_name}")
            print("Available datasets:")
            for d in ALL_DATASETS:
                print(f"  - {d.name}")
            return
    elif priority:
        if priority == 1:
            datasets = PRIORITY_1
        elif priority == 2:
            datasets = PRIORITY_2
        elif priority == 3:
            datasets = PRIORITY_3
        elif priority == 4:
            datasets = PRIORITY_4
        else:
            print(f"Invalid priority: {priority}")
            return
    else:
        datasets = ALL_DATASETS
    
    print(f"\n{'=' * 60}")
    print(f"NICTO DATA COLLECTION")
    print(f"{'=' * 60}")
    print(f"Datasets: {[d.name for d in datasets]}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Process: {process}")
    print(f"Tokenize: {tokenize}")
    print(f"{'=' * 60}\n")
    
    # Download
    print("STEP 1: Downloading datasets...")
    paths = download_all(
        datasets=datasets,
        max_samples_per_dataset=max_samples,
    )
    print(f"\nDownloaded {len(paths)} datasets")
    
    # Process
    if process:
        print("\nSTEP 2: Processing datasets...")
        from .registry import RAW_DIR, PROCESSED_DIR
        
        for path in paths:
            try:
                process_file(path, max_samples=max_samples)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        print("\nProcessing complete!")
    
    # Tokenize
    if tokenize:
        print("\nSTEP 3: Tokenizing datasets...")
        from .registry import PROCESSED_DIR
        
        processed_files = list(PROCESSED_DIR.glob("*.jsonl"))
        for path in processed_files:
            try:
                tokenize_file(path, max_samples=max_samples)
            except Exception as e:
                print(f"Error tokenizing {path}: {e}")
        
        print("\nTokenization complete!")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Downloaded: {len(paths)} datasets")
    
    if process:
        from .registry import PROCESSED_DIR
        processed_count = len(list(PROCESSED_DIR.glob("*.jsonl")))
        print(f"Processed: {processed_count} files")
    
    if tokenize:
        from .registry import TOKENIZED_DIR
        tokenized_dirs = list(TOKENIZED_DIR.iterdir())
        print(f"Tokenized: {len(tokenized_dirs)} directories")
    
    print(f"\nNext steps:")
    print(f"  1. Review processed data in: {Path(__file__).parent / 'processed'}")
    print(f"  2. Update training config with data_path")
    print(f"  3. Run training with real data")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="NICTO Data Collection")
    parser.add_argument("--priority", type=int, help="Download priority (1-4)")
    parser.add_argument("--dataset", type=str, help="Download specific dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--max-samples", type=int, help="Max samples per dataset")
    parser.add_argument("--process", action="store_true", help="Process after download")
    parser.add_argument("--tokenize", action="store_true", help="Tokenize after processing")
    parser.add_argument("--config", default="t4_medium", choices=["t4_small", "t4_medium", "t4_large", "k8s_full"])
    
    args = parser.parse_args()
    
    if args.all:
        collect_data(
            max_samples=args.max_samples,
            process=args.process,
            tokenize=args.tokenize,
            config_name=args.config,
        )
    elif args.priority or args.dataset:
        collect_data(
            priority=args.priority,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            process=args.process,
            tokenize=args.tokenize,
            config_name=args.config,
        )
    else:
        print("Specify --priority N, --dataset NAME, or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
