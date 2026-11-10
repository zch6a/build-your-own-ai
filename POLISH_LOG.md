# Polish log

Each round goes through the whole project — every chapter's claims checked against a run, the code
read top to bottom, tests, first-run experience, docs (English and Chinese), CI, and a record here —
then a local commit.

## Development

- [x] **Scope.** Eight chapters, each a module in `byoai/`, a walkthrough and a demo in `chapters/NN-*/`,
  and tests: tokenizer, autograd, GPT, inference, vector database, RAG, agent, MCP. Constraints chosen
  up front: numpy is the only dependency; nothing to download (no model weights, no datasets); no API
  key — agent and RAG chapters will run on the chapter-3 model or a deterministic stand-in, with an
  optional adapter for any OpenAI-compatible endpoint (DeepSeek included) for those who have one.
- [x] **Training text** (`byoai/corpus.py`): a seeded grammar of short stories — generated, so there is
  nothing to license and every machine trains on the same text; small and regular enough for a tiny
  transformer to learn on a CPU.
- [x] **Chapter 1 — byte-level BPE tokenizer** (`byoai/tokenizer.py`): training with word counts and
  incremental pair updates (264 merges on 400 KB in 0.02 s), encoding by merge rank with a chunk
  cache, opt-in special tokens (a document cannot inject `<|endoftext|>`), save/load. 31 tests,
  including round trips of text the tokenizer never saw (other scripts, emoji, control characters).
  The demo shows something worth knowing: on this corpus training stops after 264 merges because every
  word is already one token — `vocab_size` is an upper bound — and says so instead of hiding it.
- [x] **Chapter 2 — autograd** (`byoai/autograd.py`): a `Tensor` that records its inputs and a backward closure; iterative topological sort (a 5,000-op chain does not hit the recursion limit); broadcasting undone as a sum; batched `@`, reshape, transpose, integer-array indexing with `np.add.at` (repeated token ids get both gradients), `masked_fill`, `softmax`, `log_softmax`, fused `cross_entropy` (stable at logits of ±1000), `layer_norm` built from primitives, GELU, `concatenate`; `no_grad`; AdamW (decay on matrices only); `gradcheck`. 35 tests — every operation checked against finite differences. One bug found by its own test: float32 tensors got float64 gradients (reductions return a numpy *scalar*, which the constructor upcast) — it matters for chapter 3's speed. Demo: hand-derived vs. autograd gradients, a finite-difference check, a 2-64-64-3 network learning three spirals to 100% in about a second, drawn as an ASCII map.
- [ ] Chapters 3–8.
