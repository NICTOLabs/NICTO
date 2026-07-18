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
        description="NICTO AI — neural architecture",
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
    chat_p = sub.add_parser("chat", help="Chat with NICTO (LLM + tools)")
    chat_p.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                        help="LLM provider (default: openai)")
    chat_p.add_argument("--model", default=None, help="Model name (e.g., gpt-4o-mini)")
    chat_p.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY env)")

    # tool
    tool_p = sub.add_parser("tool", help="Invoke a tool directly")
    tool_p.add_argument("name", nargs="?", help="Tool name")
    tool_p.add_argument("params", nargs="*", help="Parameters (key=value)")
    tool_p.add_argument("--list", action="store_true", help="List available tools")

    # info
    sub.add_parser("info", help="Show model info")

    # workflow
    wf_p = sub.add_parser("workflow", help="Run a multi-tool workflow")
    wf_p.add_argument("name", nargs="?", help="Workflow name (e.g. search_and_summarize)")
    wf_p.add_argument("input", nargs="?", help="Input text for the workflow")
    wf_p.add_argument("--list", action="store_true", help="List available workflows")
    wf_p.add_argument("--param", nargs="*", help="Additional parameters (key=value)")

    # knowledge
    kb_p = sub.add_parser("knowledge", help="Manage knowledge base")
    kb_p.add_argument("action", nargs="?", choices=["query", "crawl", "ingest", "stats"], help="Action to perform")
    kb_p.add_argument("value", nargs="?", help="Query text, URL to crawl, or text to ingest")
    kb_p.add_argument("--top-k", type=int, default=5, help="Number of results (query)")
    kb_p.add_argument("--title", default="", help="Title (ingest)")
    kb_p.add_argument("--url", default="", help="Source URL (ingest)")

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
        from nicto_ai.chat import run_chat
        run_chat(backend=args.backend, model=args.model)

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

    elif args.command == "workflow":
        from nicto_ai.tools import create_default_registry
        from nicto_ai.agent import ToolAgent, WorkflowEngine

        registry = create_default_registry()
        agent = ToolAgent(registry)
        engine = WorkflowEngine(agent)

        if args.list or not args.name:
            print("Available workflows:")
            for wf in engine.list_presets():
                print(f"  {wf['name']}: {wf['description']}")
            return

        wf = engine._presets.get(args.name)
        if not wf:
            print(f"Unknown workflow: {args.name}")
            print("Available:", [w["name"] for w in engine.list_presets()])
            return

        input_data = {"input": args.input or ""}
        if args.param:
            for p in args.param:
                if "=" in p:
                    k, v = p.split("=", 1)
                    input_data[k] = v

        result = engine.run(wf, input_data)
        print(f"Workflow: {result.name}")
        print(f"Success: {result.success}")
        print(f"Time: {result.elapsed_ms:.0f}ms")
        print()
        for name, step in result.steps.items():
            status = step["status"]
            print(f"  [{status}] {name}: {step.get('tool', '')}")
            if status == "success" and "result" in step:
                result_text = step["result"]
                if isinstance(result_text, str) and len(result_text) > 200:
                    result_text = result_text[:200] + "..."
                print(f"    -> {result_text}")

    elif args.command == "knowledge":
        from nicto_ai.tools import KnowledgeTool
        tool = KnowledgeTool()

        if args.action == "stats":
            result = tool._execute("stats")
        elif args.action == "crawl":
            if not args.value:
                print("Usage: nicto knowledge crawl <URL>")
                return
            result = tool._execute(f"crawl:{args.value}")
        elif args.action == "ingest":
            if not args.value:
                print("Usage: nicto knowledge ingest <text> [--title TITLE] [--url URL]")
                return
            result = tool._execute(args.value, title=args.title, url=args.url)
            if result.success:
                result._execute = "ingested"  # override for output
        else:
            if not args.value:
                print("Usage: nicto knowledge query <text> [--top-k N]")
                return
            result = tool._execute(args.value, top_k=args.top_k)

        if result.success:
            output = result.output
            if isinstance(output, list):
                print(f"Found {len(output)} results ({result.metadata.get('query_time_ms', 0):.0f}ms):")
                for i, r in enumerate(output, 1):
                    print(f"  {i}. {r.get('title', 'Untitled')} (score: {r.get('score', 0):.2f})")
                    if r.get("url"):
                        print(f"     URL: {r['url']}")
                    if r.get("summary"):
                        print(f"     {r['summary'][:150]}")
                    print()
            elif isinstance(output, dict):
                for k, v in output.items():
                    print(f"  {k}: {v}")
        else:
            print(f"Error: {result.error}")

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
