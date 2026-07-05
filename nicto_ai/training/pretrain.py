"""
NICTO AI Pre-training Script
Run this to start training the model
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nicto_ai.core.model import NICTOModel
from nicto_ai.core.vision import VGG16ForNICTO
from nicto_ai.training.trainer import NICTOTrainer, TextDataset
from torch.utils.data import DataLoader


def create_model(use_small: bool = False):
    """Create NICTO model"""
    if use_small:
        print("Creating small model for testing...")
        model = NICTOModel(
            vocab_size=32000,
            dim=2048,
            max_seq_len=8192,
        )
    else:
        print("Creating full NICTO model...")
        model = NICTOModel(
            vocab_size=128000,
            dim=8192,
            max_seq_len=10000000,
        )

    # Add VGG16 vision encoder
    print("Adding VGG16 vision encoder...")
    model.vgg16 = VGG16ForNICTO(nicto_dim=8192, pretrained=True)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated size: {total_params * 4 / 1e9:.1f} GB (FP32) or {total_params * 2 / 1e9:.1f} GB (FP16)")

    return model


def get_training_config(args):
    """Get training configuration"""
    return {
        # Optimization
        "learning_rate": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,

        # Schedule
        "warmup_steps": args.warmup_steps,
        "total_steps": args.total_steps,

        # Batch size
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "max_seq_len": args.seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="NICTO AI Pre-training")

    # Model
    parser.add_argument("--small", action="store_true", help="Use small model for testing")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation data")

    # Training
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--total_steps", type=int, default=1000000, help="Total training steps")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--validate_every", type=int, default=500, help="Validate every N steps")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    args = parser.parse_args()

    print("=" * 60)
    print("NICTO AI - Pre-training")
    print("=" * 60)

    # Create model
    model = create_model(use_small=args.small)

    # Resume if specified
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint.get("global_step", 0)

    # Create dataset
    print(f"Loading data from {args.data_path}...")
    train_dataset = TextDataset(args.data_path, seq_len=args.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation loader
    val_loader = None
    if args.val_data_path:
        val_dataset = TextDataset(args.val_data_path, seq_len=args.seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Training config
    config = get_training_config(args)

    # Create trainer
    trainer = NICTOTrainer(
        model=model,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Resume trainer state
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_every=args.save_every,
        validate_every=args.validate_every,
    )


if __name__ == "__main__":
    main()
