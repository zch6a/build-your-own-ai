# Changelog

## Unreleased

- Chapter 1: byte-level BPE tokenizer (`byoai/tokenizer.py`), walkthrough, demo and tests.
- Chapter 2: autograd (`byoai/autograd.py`) — reverse-mode autodiff on numpy arrays, AdamW, gradient checking; walkthrough, demo and tests.
- Chapter 3: GPT (`byoai/gpt.py`) — a decoder-only transformer trained on the stories; `byoai/zoo.py` trains and caches the shared story model.
- Chapter 4: inference (`byoai/generate.py`) — a numpy engine with a KV cache, sampling (temperature, top-k, top-p), streaming.
- Chapter 5: vector database (`byoai/vectordb.py`) — exact `FlatIndex`, `HNSWIndex` with the paper's neighbour heuristic, soft deletes, save/load; `Engine.embed` for text vectors; a demo comparing model embeddings with a bag of words.
- Chapter 3: `longer.py` trains for 2,500 steps and shows the copying circuit forming in a sudden jump.
- Autograd: float32 stays float32 end to end (constants take the tensor's dtype), 2-D matrix products, faster GELU; `gradcheck` compares tiny gradients absolutely.
- A generated training corpus (`byoai/corpus.py`).
