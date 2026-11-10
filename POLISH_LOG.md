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
- [x] **Chapter 3 — GPT** (`byoai/gpt.py`, `byoai/zoo.py`): GPT-2's architecture (token + position embeddings, pre-norm blocks, multi-head causal attention, GELU MLP, residuals, weight tying, GPT-2 initialization), AdamW with warmup + cosine and gradient clipping; the shared story model is trained once and cached by a key of everything that determines it. 392k parameters train in 102 s on a CPU to held-out loss 0.62 (untrained 6.30); stories are new (checked sentence by sentence against the training text). Tests: loss at init ≈ ln V (a first version used targets = inputs, and an untrained tied model leans toward its input token — the test was wrong, not the model), causality, a full gradient check of every parameter (tiny gradients ~1e-7 now compared absolutely: finite-difference noise), memorizing a sequence, weight tying, save/load.
  - **Speed, profiled twice**: 290 ms/step at first. A constant like `1/sqrt(D)` wrapped as a `Tensor` became a 0-d *float64 array*, which NumPy 2 promotes float32 to (a bare Python number it would not) — most of the pass ran in double precision; constants now take the tensor's dtype. `(B,T,C) @ (C,3C)` became one 2-D BLAS product with the weight gradient summed directly. Plus no float powers in GELU/sqrt, no gradient copies (clipping now makes new arrays). 1.47 s → 0.32 s/step for the largest size tried.
  - **What it does not learn (measured)**: the second sentence names the first sentence's animal in 22% of stories after 500 steps, 9% after 1,000 — chance is 10%. The loss plateaus at ~0.61; copying would be worth ~0.14 nats/token. Written into the chapter as what is measured, why (copying needs a two-head circuit that forms late), and an exercise.
- [x] **Chapter 4 — inference** (`byoai/generate.py`): a numpy engine on the trained weights with a KV cache (prefill, then decode one position at a time; rebuilt from the recent half when the 64-position window fills), sampling (temperature, top-k, top-p), stop tokens, streaming. The cached logits equal a full forward pass at every step (1e-5); greedy matches chapter 3. Demo, measured: 10.7× faster with the cache (122 → 1,317 tokens/s); greedy writes the same story every time (2% distinct); temperature 0.5–1.0 is fluent (86–88% of sentences valid in the grammar), 1.5 breaks (68%), 2.5 is nonsense (0%); top-k / top-p bring temperature 1.5 back to 85%.
- [ ] Chapters 5–8.
