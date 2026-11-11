# Changelog

## 0.1.0 — first release (not yet published)

Eight chapters, each a module in `byoai/`, a walkthrough and a demo in `chapters/`, and tests. Python 3.10+,
numpy as the only dependency, nothing to download, no API key.

**The stack**

- **Corpus** (`corpus.py`): a seeded grammar of short stories for training, and 60 encyclopedia articles about
  invented towns with 360 questions for retrieval and the agent — generated, so there is nothing to license.
- **1 · Tokenizer** (`tokenizer.py`): byte-level BPE — training with incremental pair counts, encoding by merge
  rank with a cache, special tokens that text cannot inject, save/load. Lossless on any input.
- **2 · Autograd** (`autograd.py`): a `Tensor` with reverse-mode differentiation — broadcasting, batched matrix
  products, indexing, masking, fused softmax/cross-entropy, layer norm, GELU — AdamW and a finite-difference
  `gradcheck` for every operation.
- **3 · GPT** (`gpt.py`, `zoo.py`): GPT-2's architecture, trained on a laptop CPU in about two minutes; the
  shared model is cached by its settings and source code. `longer.py` shows the induction-head phase change.
- **4 · Inference** (`generate.py`): a numpy engine with a KV cache (about 10× faster), temperature, top-k,
  top-p, streaming, and sentence embeddings.
- **5 · Vector database** (`vectordb.py`): exact search and HNSW with the paper's neighbour heuristic, soft
  deletes and save/load; recall measured against exact search.
- **6 · RAG** (`rag.py`, `llm.py`): sentence-aware chunking with title headers, BM25, LSA, reciprocal rank
  fusion, hit@k/MRR, a prompt builder, a stand-in reader that cites or says "I don't know" (0% wrong
  answers), and an optional client for any OpenAI-compatible endpoint (DeepSeek included), off unless set.
- **7 · Agent** (`agent.py`, `planner.py`): a tool-calling loop with schemas from type hints, validation,
  errors fed back to the model, step and repeat limits, result truncation, a safe calculator and a
  sandboxed file reader; an offline stand-in model with tasks and a noisy-model wrapper.
- **8 · MCP** (`mcp.py`): a Model Context Protocol server and client over stdio — handshake with version
  negotiation (2024-11-05 to 2025-11-25), tools/list with pagination, tools/call, cancellation, graceful
  shutdown — and `python -m byoai.mcp`, a working server for Claude Code or any MCP client.

**Quality**

- 238 tests (including doctests) and every demo, passing locally on Windows with Python 3.14; CI is set up for Linux, Windows
  and macOS with Python 3.10 and 3.13 (it runs once the repository is on GitHub).
- Every number in the walkthroughs was produced by running the code; several first drafts were corrected
  when the measurement disagreed (see [POLISH_LOG.md](POLISH_LOG.md)).
- Five rounds of review: an independent code review, a learner doing the exercises, a check against the
  MCP/OpenAI/DeepSeek specifications and the original papers, mutation testing (29 deliberate bugs, all
  caught), and a final read-through.
