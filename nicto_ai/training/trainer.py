"""
NICTO AI Training Pipeline
Complete training system for all 6 networks
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


class CosineWarmupScheduler:
    """Learning rate scheduler with warmup"""

    def __init__(self, optimizer, warmup_steps: int, max_lr: float, min_lr: float, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.max_lr * (self.current_step / self.warmup_steps)
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class TextDataset(Dataset):
    """Dataset for text pretraining"""

    def __init__(self, data_path: str, seq_len: int = 2048, vocab_size: int = 128000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if os.path.isdir(data_path):
            self.files = list(Path(data_path).glob("*.bin"))
            self.data = None
        else:
            self.files = None
            self.data = torch.load(data_path)

    def __len__(self):
        if self.data is not None:
            return max(0, len(self.data) - self.seq_len - 1)
        return len(self.files) * 1000  # Estimate

    def __getitem__(self, idx):
        if self.data is not None:
            start = idx
            end = start + self.seq_len + 1
            chunk = self.data[start:end]
            return {
                "input_ids": chunk[:-1].long(),
                "labels": chunk[1:].long(),
            }
        else:
            file_idx = idx // 1000
            with open(self.files[file_idx], 'rb') as f:
                data = torch.load(f)
            start = idx % max(1, len(data) - self.seq_len - 1)
            chunk = data[start:start + self.seq_len + 1]
            return {
                "input_ids": chunk[:-1].long(),
                "labels": chunk[1:].long(),
            }


class MultimodalDataset(Dataset):
    """Dataset for multimodal training (text + images)"""

    def __init__(self, image_paths: list, captions: list, seq_len: int = 2048, image_size: int = 224):
        self.image_paths = image_paths
        self.captions = captions
        self.seq_len = seq_len
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = transform(image)

        caption = self.captions[idx]

        return {
            "image": image,
            "caption": caption,
        }


class NICTOTrainer:
    """
    Complete training pipeline for NICTO AI

    Phases:
    1. Pre-training (next token prediction)
    2. Network-specific fine-tuning
    3. Integration training
    4. Alignment (RLHF)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        output_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.scaler = GradScaler()  # Mixed precision

        # Setup optimizer
        self._setup_optimizer()

        # Setup logging
        self.log_file = self.output_dir / "training_log.json"
        self.logs = []

    def _setup_optimizer(self):
        """Setup optimizer with weight decay"""
        config = self.config

        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': config.get('weight_decay', 0.1)},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.get('learning_rate', 3e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.95)),
            eps=config.get('eps', 1e-8),
        )

        # Learning rate scheduler
        total_steps = config.get('total_steps', 1000000)
        warmup_steps = config.get('warmup_steps', 2000)
        max_lr = config.get('learning_rate', 3e-4)
        min_lr = config.get('min_lr', 1e-5)

        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, max_lr, min_lr, total_steps
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training loss"""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        with autocast(dtype=torch.bfloat16):
            outputs = self.model(input_ids, labels=labels)

            # Main loss
            loss = outputs["loss"]

            # Auxiliary losses
            aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=self.device))

            # Total loss
            total_loss = loss + 0.01 * aux_loss

        return {
            "loss": total_loss,
            "main_loss": loss,
            "aux_loss": aux_loss,
            "logits": outputs["logits"],
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        # Compute loss
        loss_dict = self.compute_loss(batch)

        # Backward pass with mixed precision
        self.scaler.scale(loss_dict["loss"]).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.global_step += 1

        return {
            "loss": loss_dict["loss"].item(),
            "main_loss": loss_dict["main_loss"].item(),
            "aux_loss": loss_dict["aux_loss"].item(),
            "lr": self.scheduler._get_lr(),
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        total_main_loss = 0
        n_batches = 0

        for batch in val_loader:
            loss_dict = self.compute_loss(batch)
            total_loss += loss_dict["loss"].item()
            total_main_loss += loss_dict["main_loss"].item()
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_main_loss": total_main_loss / max(n_batches, 1),
        }

    def save_checkpoint(self, name: str = "latest"):
        """Save model checkpoint"""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        print(f"Loaded checkpoint from step {self.global_step}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics"""
        metrics["global_step"] = self.global_step
        metrics["epoch"] = self.epoch
        metrics["timestamp"] = time.time()
        self.logs.append(metrics)

        # Save logs
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

        # Print metrics
        if self.global_step % 100 == 0:
            print(f"Step {self.global_step}: " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 1,
        save_every: int = 1000,
        validate_every: int = 500,
    ):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Total steps: {len(train_loader) * num_epochs}")
        print(f"Device: {self.device}")
        print("=" * 60)

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            for batch in train_loader:
                # Training step
                metrics = self.train_step(batch)
                self.log_metrics(metrics)

                # Validation
                if self.global_step % validate_every == 0 and val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    self.log_metrics(val_metrics)

                    # Save best model
                    if val_metrics["val_loss"] < self.best_loss:
                        self.best_loss = val_metrics["val_loss"]
                        self.save_checkpoint("best")

                # Save checkpoint
                if self.global_step % save_every == 0:
                    self.save_checkpoint("latest")

        # Final save
        self.save_checkpoint("final")
        print("\nTraining complete!")


class NICTOTrainerDistributed(NICTOTrainer):
    """
    Distributed training with DDP or FSDP

    Usage:
        torchrun --nproc_per_node=4 train.py
    """

    def __init__(self, *args, fsdp: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.fsdp = fsdp

        if self.world_size > 1:
            self._setup_distributed()

    def _setup_distributed(self):
        """Setup distributed process group and wrap model."""
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.model = self.model.to(self.device)

        if self.fsdp:
            self._wrap_fsdp()
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Re-setup optimizer after model wrapping
        self._setup_optimizer()

    def _wrap_fsdp(self):
        """Wrap model with Fully Sharded Data Parallel for large models."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            device_id=self.local_rank,
            sync_module_states=True,
            use_orig_params=True,
        )

    def _get_state_dict(self):
        """Extract state dict from DDP/FSDP wrapper."""
        if self.fsdp:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                return self.model.state_dict()
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def save_checkpoint(self, name: str = "latest"):
        """Save checkpoint (only on rank 0)."""
        if self.local_rank != 0:
            return

        if self.fsdp:
            state_dict = self._get_state_dict()
        else:
            state_dict = self.model.state_dict()

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)
        print(f"[rank 0] Saved checkpoint: {path}")

    def cleanup(self):
        """Clean up distributed process group."""
        if self.world_size > 1:
            torch.distributed.destroy_process_group()
