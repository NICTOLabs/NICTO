"""NICTO AI - CLI entry point"""

import argparse
import sys


def print_banner():
    print("  _   _ ___ ___  ____ ___  ")
    print(" | \\ | |_ _/ _ \\/ ___/ _ \\ ")
    print(" |  \\| | | | | | |  | | | |")
    print(" | |\\  | | | |_| | |__| |_| |")
    print(" |_| \\_|___|\\___/\\____\\___/ ")
    print()


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
    chat_p = sub.add_parser("chat", help="Chat with NICTO (tool-aware)")
    chat_p.add_argument("--checkpoint", default="nicto_model_final.pt", help="Model checkpoint")
    chat_p.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    chat_p.add_argument("--model", action="store_true", help="Force LLM mode (no tool detection)")
    chat_p.add_argument("--tools", action="store_true", help="List available tools and exit")

    # tool
    tool_p = sub.add_parser("tool", help="Invoke a tool directly")
    tool_p.add_argument("name", nargs="?", help="Tool name")
    tool_p.add_argument("params", nargs="*", help="Parameters (key=value)")
    tool_p.add_argument("--list", action="store_true", help="List available tools")

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
        print_banner()
        print("NICTO AI - Chat (tool-aware)")
        print("  /help          - Show available commands")
        print("  /tools         - List available tools")
        print("  /tool_name ... - Invoke a tool directly")
        print("  Type naturally - NICTO auto-detects tools or uses LLM")
        print()

        from nicto_ai.voice.backend_interface import NICTOBackend, Message, EchoBackend
        from nicto_ai.tools import create_default_registry
        from nicto_ai.agent import ToolAgent

        # Load tools
        registry = create_default_registry()
        agent = ToolAgent(registry)

        # Load LLM backend
        backend = None
        if not args.model and not args.tools:
            backend = NICTOBackend(checkpoint_path=args.checkpoint)
            if not backend.is_available():
                print("(Model not loaded - tools-only mode)\n")

        history = []

        def llm_chat(user_input: str) -> str:
            """Fall back to LLM"""
            if backend and backend.is_available():
                history.append(Message(role="user", content=user_input))
                result = backend.complete(history, max_tokens=args.max_tokens)
                history.append(Message(role="assistant", content=result.text))
                return result.text
            return None

        # Print available tools on demand
        if args.tools:
            print(agent.list_tools())
            return

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

            # Help
            if user_input == "/help":
                print("Commands:")
                print("  /tools            - List available tools")
                print("  /tool_name params  - Invoke tool directly (e.g. /calculator 2+2)")
                print("  /model text       - Force LLM generation")
                print("  /clear            - Clear history")
                print("  quit              - Exit")
                print()
                continue

            if user_input == "/clear":
                history.clear()
                print("(History cleared)\n")
                continue

            # Force model mode
            if user_input.startswith("/model "):
                text = user_input[7:]
                resp = llm_chat(text)
                if resp:
                    print(f"NICTO: {resp}\n")
                else:
                    print("(Model not available. Use a tool instead.)\n")
                continue

            # Tool list
            if user_input == "/tools":
                print(agent.list_tools())
                print()
                continue

            # Try tool agent first
            response = agent.process(user_input)

            if response:
                if response.tool_result and response.tool_result.execution_time_ms > 0:
                    print(f"[{response.tool_name} in {response.execution_time_ms:.0f}ms]")
                print(f"NICTO: {response.text}\n")
            else:
                # Fall back to LLM
                resp = llm_chat(user_input)
                if resp:
                    print(f"NICTO: {resp}\n")
                else:
                    print("(No tool matched and model not available. Try /tools to see what I can do.)\n")

    elif args.command == "tool":
        from nicto_ai.tools import create_default_registry
        from nicto_ai.agent import ToolAgent

        registry = create_default_registry()
        agent = ToolAgent(registry)

        if args.list or not args.name:
            print(agent.list_tools())
            return

        # Build params dict
        params = {}
        for p in args.params:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v

        # Invoke via agent format
        cmd_text = f"/{args.name} " + " ".join(f"{k}={v}" for k, v in params.items())
        response = agent.process(cmd_text)
        if response:
            print(response.text)
        else:
            print(f"Unknown tool: {args.name}")

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
        print()
        print("Available tools:")
        from nicto_ai.tools import create_default_registry
        for t in create_default_registry().list_tools():
            print(f"  {t['name']}: {t['description'][:70]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
