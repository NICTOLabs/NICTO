"""
NICTO Configuration System.

YAML-based configuration with:
  - Model configs (dimensions, layers, heads)
  - Training configs (lr, batch_size, epochs)
  - Generation configs (steps, temperature, etc.)
  - API configs (host, port, auth)
  - Environment-specific overrides
  - Config validation and merging
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    yaml = None


# ==============================================================================
# Config Base
# ==============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    latent_dim: int = 512
    hidden_dim: int = 2048
    num_heads: int = 16
    num_layers: int = 24
    vocab_size: int = 32000
    max_seq_len: int = 4096
    dropout: float = 0.1
    activation: str = "silu"


@dataclass
class GenerationConfig:
    """Generation configuration."""
    num_steps: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    cfg_scale: float = 7.5
    num_points_3d: int = 2048
    video_frames: int = 16
    audio_duration: float = 5.0
    batch_size: int = 1


@dataclass
class TrainingConfig:
    """Training configuration."""
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 32
    grad_accumulation: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 10


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_batch_size: int = 32
    timeout_s: int = 300


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    data_dir: str = "data"
    cache_dir: str = "cache"
    max_seq_len: int = 4096
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class NictoConfig:
    """Main NICTO configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Global settings
    device: str = "auto"
    seed: int = 42
    debug: bool = False
    log_level: str = "INFO"

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ==============================================================================
# Config Loader
# ==============================================================================

class ConfigLoader:
    """Loads and merges YAML configurations."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        if yaml is None:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def save_yaml(self, data: Dict[str, Any], path: Path):
        """Save to YAML file."""
        if yaml is None:
            raise ImportError("PyYAML required")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load_config(self, name: str = "default") -> NictoConfig:
        """Load a named configuration."""
        if name in self._cache:
            return self._cache[name]

        # Load base config
        base_path = self.config_dir / f"{name}.yaml"
        if base_path.exists():
            data = self.load_yaml(base_path)
        else:
            data = {}

        # Apply environment overrides
        env = os.environ.get("NICTO_ENV", "dev")
        env_path = self.config_dir / f"{name}.{env}.yaml"
        if env_path.exists():
            env_data = self.load_yaml(env_path)
            data = self._deep_merge(data, env_data)

        # Convert to NictoConfig
        config = self._dict_to_config(data)
        self._cache[name] = config

        return config

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _dict_to_config(self, data: dict) -> NictoConfig:
        """Convert dictionary to NictoConfig."""
        model_data = data.get("model", {})
        gen_data = data.get("generation", {})
        train_data = data.get("training", {})
        api_data = data.get("api", {})
        data_pipeline = data.get("data", {})

        return NictoConfig(
            model=ModelConfig(**{k: v for k, v in model_data.items() if k in ModelConfig.__dataclass_fields__}),
            generation=GenerationConfig(**{k: v for k, v in gen_data.items() if k in GenerationConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in train_data.items() if k in TrainingConfig.__dataclass_fields__}),
            api=APIConfig(**{k: v for k, v in api_data.items() if k in APIConfig.__dataclass_fields__}),
            data=DataConfig(**{k: v for k, v in data_pipeline.items() if k in DataConfig.__dataclass_fields__}),
            device=data.get("device", "auto"),
            seed=data.get("seed", 42),
            debug=data.get("debug", False),
            log_level=data.get("log_level", "INFO"),
        )

    def save_config(self, config: NictoConfig, name: str = "default"):
        """Save config to YAML."""
        data = {
            "model": {
                "latent_dim": config.model.latent_dim,
                "hidden_dim": config.model.hidden_dim,
                "num_heads": config.model.num_heads,
                "num_layers": config.model.num_layers,
                "vocab_size": config.model.vocab_size,
                "max_seq_len": config.model.max_seq_len,
                "dropout": config.model.dropout,
            },
            "generation": {
                "num_steps": config.generation.num_steps,
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
                "cfg_scale": config.generation.cfg_scale,
            },
            "training": {
                "lr": config.training.lr,
                "batch_size": config.training.batch_size,
                "max_steps": config.training.max_steps,
            },
            "api": {
                "host": config.api.host,
                "port": config.api.port,
            },
            "device": config.device,
            "seed": config.seed,
            "debug": config.debug,
        }

        self.save_yaml(data, self.config_dir / f"{name}.yaml")


# ==============================================================================
# Default Configs
# ==============================================================================

def get_default_config() -> NictoConfig:
    """Get the default configuration."""
    return NictoConfig()


def get_small_config() -> NictoConfig:
    """Get a small config for testing/development."""
    return NictoConfig(
        model=ModelConfig(
            latent_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
        ),
        generation=GenerationConfig(
            num_steps=10,
            num_points_3d=1024,
        ),
        training=TrainingConfig(
            batch_size=8,
            max_steps=10000,
        ),
    )


def get_large_config() -> NictoConfig:
    """Get a large config for production."""
    return NictoConfig(
        model=ModelConfig(
            latent_dim=1024,
            hidden_dim=4096,
            num_heads=32,
            num_layers=48,
        ),
        generation=GenerationConfig(
            num_steps=50,
            num_points_3d=8192,
            video_frames=32,
        ),
        training=TrainingConfig(
            batch_size=64,
            max_steps=1000000,
        ),
    )


# ==============================================================================
# Global Config
# ==============================================================================

_config: Optional[NictoConfig] = None
_loader = ConfigLoader()


def load_config(name: str = "default") -> NictoConfig:
    """Load or get cached config."""
    global _config
    if _config is None:
        _config = _loader.load_config(name)
    return _config


def set_config(config: NictoConfig):
    """Set global config."""
    global _config
    _config = config


def get_config() -> NictoConfig:
    """Get current global config."""
    global _config
    if _config is None:
        _config = get_default_config()
    return _config
