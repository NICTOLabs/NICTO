# NICTO AI

Neural architecture — reasoning, memory, emotion, and consciousness in one model.

**Status:** Undergoing training. Current checkpoint is ~207M parameters (dim=1024, 6 reasoning layers, 4 MoE experts). No benchmarks run yet.

## Quick Start

```bash
git clone https://github.com/NICTOLabs/NICTO.git
cd NICTO
pip install -e .

# Chat (tool-aware)
nicto chat

# Invoke tools directly
nicto tool web_search query="Python programming"
nicto tool calculator expression="2+2"

# List all tools
nicto info
```

## Tools (all 22 working)

NICTO has 22 built-in tools accessible via natural language or direct invocation:

| Category | Tools |
|----------|-------|
| **Knowledge** | `web_search` (Bing via TLS fingerprint), `knowledge_query` |
| **Development** | `code_executor` (sandboxed), `shell`, `code_review`, `file_manager` |
| **Content** | `content_writer`, `summarizer` (5 modes), `markdown_builder` |
| **Code** | `test_generator` (pytest/Jest), `regex_builder` |
| **Data** | `data_analysis`, `json_builder`, `text_to_sql` |
| **Analysis** | `text_analyzer`, `code_review` |
| **Math** | `calculator`, `math_engine` (derivatives/matrix/statistics) |
| **Utility** | `translator`, `api_caller`, `password_generator`, `hash_tool`, `color_palette` |

Usage: `nicto chat` — type naturally (e.g., "search for Python", "calculate 2+2", "summarize this"). Tools are auto-detected; falls back to LLM.

## C++ Acceleration Engine

The tokenizer and sampler have optional C++ bindings (compiled via MSVC 2022):

| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| Tokenizer (4K chars) | 691ms | 266ms | **2.6x** |
| Sampler (vocab 32K) | 3157ms | 1909ms | **1.7x** |
| Sampler (vocab 256) | 259ms | 28ms | **9.4x** |

Build with: `build.bat` (requires VS Build Tools 2022).

## Architecture

NICTO combines multiple neural approaches in a single model:

| Component | What it does |
|-----------|-------------|
| **Reasoning** (MoE) | 128-head attention + 64-expert MoE + FFN for logic, math, code |
| **Memory** (SSM) | 48-layer Mamba SSM + hierarchical memory for long context |
| **Emotion** (LNN) | 20-layer Liquid Neural Net for adaptive behavior |
| **Creativity** | 12-layer Transformer Encoder for generation |
| **Perception** | Multihead attention for multimodal input |
| **Consciousness** | Meta-cognitive projection layer for self-monitoring |
| **Neural Bus** | Cross-network attention + fusion gate + NetworkPriorityGate |

Full 150B design requires ~200 A100s. The 207M checkpoint runs on consumer GPUs.

## Voice System

STT → LLM/ToolAgent → TTS pipeline. Supports sandboxed code execution and tool-aware responses.

## License

**All rights reserved.** Copyright © NICTOLabs. No part of this software, its source code, architecture, models, trained weights, or associated ideas may be copied, reproduced, modified, distributed, redistributed, published, or used — in whole or in part — without the express prior written permission of NICTOLabs. This software is made available solely for the private use of its owners and is not licensed for any other purpose.
