# Changelog

## Unreleased

- Chapter 1: byte-level BPE tokenizer (`byoai/tokenizer.py`), walkthrough, demo and tests.
- Chapter 2: autograd (`byoai/autograd.py`) — reverse-mode autodiff on numpy arrays, AdamW, gradient checking; walkthrough, demo and tests.
- Chapter 3: GPT (`byoai/gpt.py`) — a decoder-only transformer trained on the stories; `byoai/zoo.py` trains and caches the shared story model.
- Chapter 4: inference (`byoai/generate.py`) — a numpy engine with a KV cache, sampling (temperature, top-k, top-p), streaming.
- Chapter 5: vector database (`byoai/vectordb.py`) — exact `FlatIndex`, `HNSWIndex` with the paper's neighbour heuristic, soft deletes, save/load; `Engine.embed` for text vectors; a demo comparing model embeddings with a bag of words.
- Chapter 6: RAG (`byoai/rag.py`) — sentence-aware chunking with title headers, BM25, LSA, reciprocal rank fusion, retrieval metrics, a prompt builder and a literal stand-in reader; `knowledge_base()` in `corpus.py`; `byoai/llm.py`, an optional client for OpenAI-compatible endpoints (DeepSeek included), off unless configured.
- Chapter 7: agent (`byoai/agent.py`) — `@tool` schemas from type hints, argument validation, errors fed back to the model, step limit, repeated-call guard, result truncation, `safe_eval` and a sandboxed file reader; `byoai/planner.py`, an offline stand-in model with tasks, tools and a noisy-model wrapper.
- Chapter 8: MCP (`byoai/mcp.py`) — JSON-RPC 2.0 over stdio, initialize/version negotiation, tools/list, tools/call with protocol vs tool errors; a subprocess client with timeouts whose tools plug into the agent; `python -m byoai.mcp` serves the town tools.
- Chapter 3: `longer.py` trains for 2,500 steps and shows the copying circuit forming in a sudden jump.
- Autograd: float32 stays float32 end to end (constants take the tensor's dtype), 2-D matrix products, faster GELU; `gradcheck` compares tiny gradients absolutely.
- A generated training corpus (`byoai/corpus.py`).
