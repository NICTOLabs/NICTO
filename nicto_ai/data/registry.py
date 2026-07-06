"""
NICTO Dataset Registry
=====================
Curated list of datasets for training NICTO AI.
Each dataset has: name, HuggingFace ID, split, field mapping, license, and priority.

Usage:
    python -m nicto_ai.data.registry
    
This will download and prepare all Priority 1 datasets.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# Data directories
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZED_DIR = DATA_DIR / "tokenized"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, TOKENIZED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    hf_id: str
    split: str
    text_field: str
    license: str
    priority: int  # 1=highest, 5=lowest
    tokens_approx: str  # Approximate token count
    description: str
    filter_field: Optional[str] = None
    filter_value: Optional[str] = None
    min_length: int = 100  # Minimum text length in chars
    max_length: int = 100000  # Maximum text length in chars
    subset: Optional[str] = None  # For datasets with subsets


# =============================================================================
# PRIORITY 1: Core Pretraining Data
# =============================================================================

FINWEB_EDU = DatasetConfig(
    name="FineWeb-Edu",
    hf_id="HuggingFaceFW/fineweb-edu",
    split="train",
    text_field="text",
    license="ODC-By",
    priority=1,
    tokens_approx="1.3T",
    description="Educational quality web text. Best for knowledge and reasoning.",
    min_length=200,
)

SLIMPAJAMA = DatasetConfig(
    name="SlimPajama",
    hf_id="cerebras/SlimPajama-627B",
    split="train",
    text_field="text",
    license="Apache 2.0",
    priority=1,
    tokens_approx="627B",
    description="Cleaned and deduplicated version of RedPajama.",
    min_length=100,
)

WIKIPEDIA = DatasetConfig(
    name="Wikipedia",
    hf_id="wikimedia/wikipedia",
    split="20231101.en",
    text_field="text",
    license="CC-BY-SA-3.0",
    priority=1,
    tokens_approx="20B",
    description="English Wikipedia articles. Encyclopedia knowledge.",
    min_length=500,
)


# =============================================================================
# PRIORITY 2: Code & Math
# =============================================================================

THE_STACK_PYTHON = DatasetConfig(
    name="The Stack (Python)",
    hf_id="bigcode/thestack",
    split="train",
    text_field="content",
    license="MIT",
    priority=2,
    tokens_approx="50B",
    description="Python code from GitHub. Teaches code structure.",
    filter_field="lang",
    filter_value="Python",
    min_length=200,
    subset="Python",
)

OPENHERMES = DatasetConfig(
    name="OpenHermes 2.5",
    hf_id="teknium/OpenHermes-2.5",
    split="train",
    text_field="conversations",  # Needs conversion
    license="OpenAI",
    priority=2,
    tokens_approx="1B",
    description="High-quality instruction following data.",
)

GSM8K = DatasetConfig(
    name="GSM8K",
    hf_id="openai/gsm8k",
    split="train",
    text_field="question",
    license="MIT",
    priority=2,
    tokens_approx="10M",
    description="Grade school math problems with chain-of-thought.",
)

MATH = DatasetConfig(
    name="MATH",
    hf_id="hendrycks/competition_math",
    split="train",
    text_field="problem",
    license="MIT",
    priority=2,
    tokens_approx="100M",
    description="Competition math problems.",
)


# =============================================================================
# PRIORITY 3: Knowledge & Reasoning
# =============================================================================

ULTRACHAT = DatasetConfig(
    name="UltraChat",
    hf_id="stingning/ultrachat",
    split="train",
    text_field="prompt",  # Needs conversion
    license="MIT",
    priority=3,
    tokens_approx="5B",
    description="Multi-turn conversations for chat ability.",
)

DOLLY = DatasetConfig(
    name="Dolly 15K",
    hf_id="databricks/dolly-15k",
    split="train",
    text_field="instruction",
    license="CC-BY-SA-3.0",
    priority=3,
    tokens_approx="10M",
    description="Human-written instruction following data.",
)

PUBMED = DatasetConfig(
    name="PubMed",
    hf_id="pubmed_qa",
    split="train",
    text_field="question",
    license="CC0",
    priority=3,
    tokens_approx="200M",
    description="Medical abstracts and questions.",
)


# =============================================================================
# PRIORITY 4: Evaluation Benchmarks
# =============================================================================

MMLU = DatasetConfig(
    name="MMLU",
    hf_id="cais/mmlu",
    split="test",
    text_field="question",
    license="MIT",
    priority=4,
    tokens_approx="50M",
    description="Massive Multitask Language Understanding benchmark.",
)

ARC = DatasetConfig(
    name="ARC",
    hf_id="allenai/arc",
    split="test",
    text_field="question",
    license="CC-BY-4.0",
    priority=4,
    tokens_approx="10M",
    description="AI2 Reasoning Challenge.",
)

HELLASWAG = DatasetConfig(
    name="HellaSwag",
    hf_id="Rowan/hellaswag",
    split="validation",
    text_field="activity_label",
    license="MIT",
    priority=4,
    tokens_approx="10M",
    description="Commonsense natural language inference.",
)

TRUTHFULQA = DatasetConfig(
    name="TruthfulQA",
    hf_id="truthfulqa/truthful_qa",
    split="validation",
    text_field="question",
    license="MIT",
    priority=4,
    tokens_approx="5M",
    description="Measuring truthfulness in language models.",
)


# =============================================================================
# DATASET REGISTRIES
# =============================================================================

# All datasets
ALL_DATASETS: List[DatasetConfig] = [
    FINWEB_EDU,
    SLIMPAJAMA,
    WIKIPEDIA,
    THE_STACK_PYTHON,
    OPENHERMES,
    GSM8K,
    MATH,
    ULTRACHAT,
    DOLLY,
    PUBMED,
    MMLU,
    ARC,
    HELLASWAG,
    TRUTHFULQA,
]

# Datasets by priority
PRIORITY_1 = [d for d in ALL_DATASETS if d.priority == 1]
PRIORITY_2 = [d for d in ALL_DATASETS if d.priority == 2]
PRIORITY_3 = [d for d in ALL_DATASETS if d.priority == 3]
PRIORITY_4 = [d for d in ALL_DATASETS if d.priority == 4]

# Recommended training mix (tokens)
TRAINING_MIX = {
    "FineWeb-Edu": {"tokens": 3_000_000_000, "pct": 30},
    "SlimPajama": {"tokens": 2_000_000_000, "pct": 20},
    "Wikipedia": {"tokens": 1_000_000_000, "pct": 10},
    "The Stack (Python)": {"tokens": 1_000_000_000, "pct": 10},
    "OpenHermes 2.5": {"tokens": 1_000_000_000, "pct": 10},
    "MATH": {"tokens": 1_000_000_000, "pct": 10},
    "UltraChat": {"tokens": 1_000_000_000, "pct": 10},
}

# Evaluation datasets
EVAL_DATASETS = [MMLU, ARC, HELLASWAG, TRUTHFULQA, GSM8K]


def get_dataset_by_name(name: str) -> Optional[DatasetConfig]:
    """Get dataset config by name."""
    for d in ALL_DATASETS:
        if d.name == name:
            return d
    return None


def list_datasets_by_priority(priority: int) -> List[DatasetConfig]:
    """List all datasets of a given priority."""
    return [d for d in ALL_DATASETS if d.priority == priority]


def print_registry():
    """Print all registered datasets."""
    print("\n" + "=" * 80)
    print("NICTO DATASET REGISTRY")
    print("=" * 80)
    
    for priority in [1, 2, 3, 4]:
        datasets = list_datasets_by_priority(priority)
        print(f"\n--- Priority {priority} ({len(datasets)} datasets) ---")
        for d in datasets:
            print(f"  {d.name:25s} | {d.tokens_approx:10s} | {d.license:15s} | {d.hf_id}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED TRAINING MIX (10B tokens total)")
    print("=" * 80)
    for name, info in TRAINING_MIX.items():
        print(f"  {name:25s} | {info['tokens']:>15,} tokens | {info['pct']:>3d}%")
    print("=" * 80)


if __name__ == "__main__":
    print_registry()
