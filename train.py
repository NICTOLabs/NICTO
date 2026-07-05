"""
NICTO AI - Quick Start Training Script
Simple entry point to start training
"""

import os
import sys
import torch

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from nicto_ai.core.model import NICTOModel
from nicto_ai.core.vision import VGG16ForNICTO
from nicto_ai.training.trainer import NICTOTrainer


def create_small_model():
    """Create a small model for testing/development"""
    print("Creating small NICTO model for development...")
    model = NICTOModel(
        vocab_size=32000,
        dim=2048,
        max_seq_len=8192,
    )
    model.vgg16 = VGG16ForNICTO(nicto_dim=2048, pretrained=False)
    return model


def create_full_model():
    """Create the full NICTO model"""
    print("Creating full NICTO model...")
    model = NICTOModel(
        vocab_size=128000,
        dim=8192,
        max_seq_len=10000000,
    )
    model.vgg16 = VGG16ForNICTO(nicto_dim=8192, pretrained=True)
    return model


def generate_synthetic_data(n_samples: int = 1000, seq_len: int = 2048, vocab_size: int = 32000):
    """Generate synthetic data for testing"""
    print(f"Generating {n_samples} synthetic samples...")
    data = torch.randint(0, vocab_size, (n_samples * seq_len,))
    return data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("NICTO AI - Training")
    print("=" * 60)

    # Create model
    if args.mode == "small":
        model = create_small_model()
    else:
        model = create_full_model()

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Generate synthetic data for testing
    data = generate_synthetic_data(n_samples=args.steps * args.batch_size)

    # Create simple dataset
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training config
    config = {
        "learning_rate": 1e-4,
        "min_lr": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "total_steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
    }

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = NICTOTrainer(
        model=model,
        config=config,
        output_dir=args.output_dir,
        device=device,
    )

    # Quick training loop for testing
    print(f"\nRunning {args.steps} training steps...")
    model.train()

    for step in range(args.steps):
        batch_data = data[step * args.batch_size:(step + 1) * args.batch_size]
        if len(batch_data) < args.batch_size:
            break

        input_ids = batch_data.unsqueeze(0).to(device) if batch_data.dim() == 1 else batch_data.to(device)
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward
        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.global_step += 1

        if step % 10 == 0:
            print(f"  Step {step}/{args.steps} | Loss: {loss.item():.4f} | LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")

    # Save checkpoint
    trainer.save_checkpoint("test_run")
    print("\nTraining test complete!")
    print(f"Checkpoint saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
