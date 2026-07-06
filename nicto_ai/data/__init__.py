"""
NICTO Data Module
=================
Data collection, processing, and mixing for NICTO AI training.

Usage:
    # List available datasets
    python -m nicto_ai.data.registry
    
    # Download Priority 1 datasets
    python -m nicto_ai.data.downloader --priority 1
    
    # Process all downloaded data
    python -m nicto_ai.data.processor --all
    
    # Tokenize processed data
    python -m nicto_ai.data.processor --all --tokenize
    
    # Full pipeline
    python -m nicto_ai.data.collect --priority 1 --process
"""

from .registry import (
    DatasetConfig,
    ALL_DATASETS,
    PRIORITY_1,
    PRIORITY_2,
    PRIORITY_3,
    PRIORITY_4,
    TRAINING_MIX,
    EVAL_DATASETS,
)
from .downloader import download_dataset, download_all
from .processor import process_file, tokenize_file
from .mix_config import MixConfig, T4_SMALL, T4_MEDIUM, T4_LARGE, K8S_FULL

__all__ = [
    "DatasetConfig",
    "ALL_DATASETS",
    "PRIORITY_1",
    "PRIORITY_2",
    "PRIORITY_3",
    "PRIORITY_4",
    "TRAINING_MIX",
    "EVAL_DATASETS",
    "download_dataset",
    "download_all",
    "process_file",
    "tokenize_file",
    "MixConfig",
    "T4_SMALL",
    "T4_MEDIUM",
    "T4_LARGE",
    "K8S_FULL",
]
