"""
NICTO Data Mixing Configuration
===============================
Defines how to mix different datasets for training.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from .registry import TRAINING_MIX


@dataclass
class MixConfig:
    """Configuration for data mixing."""
    
    # Total tokens to mix
    total_tokens: int = 10_000_000_000  # 10B tokens
    
    # Dataset weights for mixing
    dataset_weights: Dict[str, float] = None
    
    # Token limits per dataset (None = use weight)
    token_limits: Dict[str, int] = None
    
    # Sampling strategy
    strategy: str = "weighted"  # "weighted", "interleaved", "sequential"
    
    def __post_init__(self):
        if self.dataset_weights is None:
            # Default: use training mix from registry
            self.dataset_weights = {
                name: info["pct"] / 100.0
                for name, info in TRAINING_MIX.items()
            }
    
    def get_token_budget(self, dataset_name: str) -> int:
        """Get token budget for a dataset."""
        if self.token_limits and dataset_name in self.token_limits:
            return self.token_limits[dataset_name]
        
        if dataset_name in self.dataset_weights:
            return int(self.total_tokens * self.dataset_weights[dataset_name])
        
        return 0


# Pre-configured mixing configs

T4_SMALL = MixConfig(
    total_tokens=1_000_000_000,  # 1B tokens
    dataset_weights={
        "FineWeb-Edu": 0.30,
        "SlimPajama": 0.20,
        "Wikipedia": 0.10,
        "The Stack (Python)": 0.10,
        "OpenHermes 2.5": 0.10,
        "MATH": 0.10,
        "UltraChat": 0.10,
    },
    strategy="weighted",
)

T4_MEDIUM = MixConfig(
    total_tokens=3_000_000_000,  # 3B tokens
    dataset_weights={
        "FineWeb-Edu": 0.30,
        "SlimPajama": 0.20,
        "Wikipedia": 0.10,
        "The Stack (Python)": 0.10,
        "OpenHermes 2.5": 0.10,
        "MATH": 0.10,
        "UltraChat": 0.10,
    },
    strategy="weighted",
)

T4_LARGE = MixConfig(
    total_tokens=10_000_000_000,  # 10B tokens
    dataset_weights={
        "FineWeb-Edu": 0.30,
        "SlimPajama": 0.20,
        "Wikipedia": 0.10,
        "The Stack (Python)": 0.10,
        "OpenHermes 2.5": 0.10,
        "MATH": 0.10,
        "UltraChat": 0.10,
    },
    strategy="weighted",
)

K8S_FULL = MixConfig(
    total_tokens=100_000_000_000,  # 100B tokens
    dataset_weights={
        "FineWeb-Edu": 0.30,
        "SlimPajama": 0.20,
        "Wikipedia": 0.10,
        "The Stack (Python)": 0.10,
        "OpenHermes 2.5": 0.10,
        "MATH": 0.10,
        "UltraChat": 0.10,
    },
    strategy="weighted",
)

# All configs
ALL_CONFIGS = {
    "t4_small": T4_SMALL,
    "t4_medium": T4_MEDIUM,
    "t4_large": T4_LARGE,
    "k8s_full": K8S_FULL,
}


def get_config(name: str) -> MixConfig:
    """Get a mixing config by name."""
    return ALL_CONFIGS.get(name, T4_MEDIUM)
