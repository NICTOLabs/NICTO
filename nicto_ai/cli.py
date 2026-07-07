"""NICTO AI - CLI entry point"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="nicto",
        description="NICTO AI — open-source neural architecture",
    )
    sub = parser.add_subparsers(dest="command")

    # train
    train_p = sub.add_parser("train", help="Train the model")
    train_p.add_argument("--config", default="colab", help="Config: colab, colab_large, k8s")
    train_p.add_argument("--data-path", default=None, help="Path to training data")
    train_p.add_argument("--use-real-data", action="store_true", help="Use real datasets")
    train_p.add_argument("--resume", default=None, help="Resume from checkpoint")
    train_p.add_argument("--steps", type=int, default=None, help="Training steps")

    # chat
    chat_p = sub.add_parser("chat", help="Chat with NICTO")
    chat_p.add_argument("--checkpoint", default="nicto_model_final.pt", help="Model checkpoint")
    chat_p.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")

    # info
    sub.add_parser("info", help="Show model info")

    args = parser.parse_args()

    if args.command == "train":
        from nicto_ai.training.train import train
        from nicto_ai.training.train import NICTOTrainConfig

        config = NICTOTrainConfig()
        kwargs = {"config": args.config}
        if args.data_path:
            kwargs["data_path"] = args.data_path
        if args.use_real_data:
            kwargs["use_real_data"] = True
        if args.resume:
            kwargs["resume"] = args.resume
        if args.steps:
            kwargs["max_steps"] = args.steps

        print("NICTO AI - Training")
        print(f"Config: {args.config}")
        train(**kwargs)

    elif args.command == "chat":
        import torch
        from nicto_ai.voice.backend_interface import NICTOBackend, Message

        print("NICTO AI - Chat")
        print(f"Loading {args.checkpoint}...")

        backend = NICTOBackend(checkpoint_path=args.checkpoint)
        if not backend.is_available():
            print("ERROR: Could not load model. Check checkpoint path.")
            sys.exit(1)

        print("Ready! Type 'quit' to exit.\n")

        history = []
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if not user_input:
                continue

            history.append(Message(role="user", content=user_input))
            result = backend.complete(history, max_tokens=args.max_tokens)
            history.append(Message(role="assistant", content=result.text))

            print(f"NICTO: {result.text}\n")

    elif args.command == "info":
        import torch
        from nicto_ai.training.model_train import NICTOTrainModel, NICTOTrainConfig

        config = NICTOTrainConfig()
        model = NICTOTrainModel(config)
        params = sum(p.numel() for p in model.parameters())

        print("NICTO AI - Model Info")
        print(f"Parameters: {params:,}")
        print(f"Dim: {config.dim}")
        print(f"Reasoning layers: {config.reasoning_layers}")
        print(f"Memory layers: {config.memory_layers}")
        print(f"Emotional layers: {config.emotional_layers}")
        print(f"Creative layers: {config.creative_layers}")
        print(f"MoE experts: {config.moe_experts}")
        print(f"Vocab size: {config.vocab_size}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
