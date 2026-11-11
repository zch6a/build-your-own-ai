<div align="center">

# 🧱 Build your own AI

**Tokenizer, autograd, GPT, inference, vector database, RAG, agent and MCP — each built from scratch.**

Small, tested, readable modules that run on a laptop CPU. No API key, no GPU, no model download.

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

---

Most explanations of how LLM systems work stop at a diagram, or start with `pip install` of the thing
being explained. Here each piece is written out — a few hundred lines of Python, with tests that pin
down what it must do and a demo you can run in seconds — so you can read it, change it, and break it.

The chapters build on each other: the tokenizer from chapter 1 feeds the GPT of chapter 3, trained with
the autograd of chapter 2; chapter 4 makes it generate fast; chapter 5's index serves chapter 6's
retrieval; chapter 7's agent calls retrieval as a tool; and chapter 8 serves those tools over the
Model Context Protocol to any client — including chapter 7's agent.

```text
                 corpus.py ── generated stories, and articles about invented towns
                     │
   1 tokenizer.py ───┤  text ⇄ token ids
   2 autograd.py  ───┤  gradients
   3 gpt.py, zoo.py ─┤  a transformer, trained once and cached
   4 generate.py  ───┘  fast generation (KV cache), sampling, embeddings
                            │
   5 vectordb.py  ──────────┤  flat search, HNSW
   6 rag.py, llm.py ────────┤  chunks → BM25 / LSA / hybrid → cited answers; optional real model
   7 agent.py, planner.py ──┤  tool-calling loop; retrieval and arithmetic as tools
   8 mcp.py ────────────────┘  the same tools, served to any MCP client over stdio
```

## What you will see

Every number in the walkthroughs comes from running the code, and some of them are surprising:

- **A phase change on a laptop.** The 3-layer GPT sits on a loss plateau, keeping the story's animal no
  better than chance — then within a few hundred steps learns to copy it (19% → 70%), the
  "induction head" jump seen in large models. ([chapter 3](chapters/03-gpt/README.md))
- **Textbook claims, tested.** Without position embeddings this GPT trains to exactly the same loss
  (the causal mask leaks position); untying the output layer *lowers* the loss from 0.62 to 0.52.
  ([chapter 3](chapters/03-gpt/README.md))
- **What an embedding holds.** Searching with the GPT's hidden states finds sentences of the same
  *shape*, not the same topic: "the owl and the key" returns "It was cold and the cat was tired".
  ([chapter 5](chapters/05-vectordb/README.md))
- **One line that doubles retrieval.** Putting the document title on each chunk takes hit@5 from 0.38 to
  0.74 for small chunks. ([chapter 6](chapters/06-rag/README.md))
- **0% wrong answers.** A reader that must cite an on-topic passage or say "I don't know" — and why a
  passage that *mentions* a name is not one *about* it. ([chapter 6](chapters/06-rag/README.md))
- **Errors are information.** With 30% of a model's tool calls broken, an agent that feeds the errors
  back succeeds 99% of the time; one that stops at the first error, 54%.
  ([chapter 7](chapters/07-agent/README.md))

## Chapters

| | Chapter | You build | Status |
|---|---|---|---|
| 1 | [Tokenizer](chapters/01-tokenizer/README.md) | byte-level BPE: training, encoding, special tokens | ✅ |
| 2 | [Autograd](chapters/02-autograd/README.md) | reverse-mode automatic differentiation on arrays, Adam | ✅ |
| 3 | [GPT](chapters/03-gpt/README.md) | a decoder-only transformer, trained on a laptop in two minutes | ✅ |
| 4 | [Inference](chapters/04-inference/README.md) | a KV cache (10× faster), temperature, top-k / top-p, streaming | ✅ |
| 5 | [Vector database](chapters/05-vectordb/README.md) | exact search, then an HNSW index, with recall measured — and what an embedding holds | ✅ |
| 6 | [RAG](chapters/06-rag/README.md) | chunking, BM25, dense (LSA) and hybrid retrieval, answers that cite or say "I don't know" — all measured | ✅ |
| 7 | [Agent](chapters/07-agent/README.md) | a tool-calling loop with validation, error feedback, limits and sandboxed tools; any OpenAI-compatible model (DeepSeek included) drops in | ✅ |
| 8 | [MCP](chapters/08-mcp/README.md) | a Model Context Protocol server and client over stdio; the agent's tools, usable from Claude Code or any MCP client | ✅ |

## Run it

```bash
git clone <this repo> && cd build-your-own-ai
uv sync                                        # Python 3.10+; numpy is the only dependency
uv run pytest                                  # every chapter's tests
uv run python chapters/01-tokenizer/run.py     # a chapter's demo
```

Without uv: `pip install -e . pytest` in a virtual environment, then `python -m pytest` and
`python chapters/01-tokenizer/run.py`.

Demos take seconds, except that the first one from chapter 3 on trains the shared story model —
about two minutes on a laptop CPU, once; it is cached in `checkpoints/` (or `$BYOAI_CHECKPOINTS`).

**A real model, optionally.** Chapters 6 and 7 run offline with stand-ins. To also try them with any
OpenAI-compatible endpoint (DeepSeek, OpenAI, a local server), set `BYOAI_LLM_BASE_URL`,
`BYOAI_LLM_MODEL` and `BYOAI_LLM_API_KEY` — see [`byoai/llm.py`](byoai/llm.py). Nothing is sent
anywhere unless `BYOAI_LLM_BASE_URL` is set.

**The tools in your own agent.** `uv run python -m byoai.mcp` is a working MCP server (stdio) offering
the chapter-7 tools; add it to Claude Code with
`claude mcp add towns -- uv run --directory /path/to/build-your-own-ai python -m byoai.mcp`.

## Principles

- **From scratch, but not toy.** Each module does the real thing at small scale — byte-level BPE as
  GPT tokenizers do it, not a character table — and says where production systems differ.
- **Runs anywhere.** numpy is the only dependency. Training data is generated by
  [`byoai/corpus.py`](byoai/corpus.py) from a seed, so every machine sees the same text and there is
  nothing to download or license.
- **Tested.** The tests are part of the explanation: they state what each piece must guarantee
  (lossless round trips, special tokens that cannot be injected, deterministic training, sandboxes
  that do not leak). And they are tested themselves: 29 deliberate bugs, one at a time — a missing
  attention scale, a dropped causal mask, a disabled sandbox check — and each one makes a test fail.
- **Measured, then described.** Every number in the walkthroughs was produced by running the code; when
  a result differed from the expectation (when the phase change happens, LSA's "better" recall), the
  text says what actually happened and why.

## License

[MIT](LICENSE)
