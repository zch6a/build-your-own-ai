"""Chapter 5 demo: exact search against HNSW, and what an embedding actually captures.

    uv run python chapters/05-vectordb/run.py

Part 3 uses the story model from chapter 3 (trained on first use, then loaded from checkpoints/).
"""

from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np

from byoai.corpus import stories
from byoai.generate import Engine
from byoai.vectordb import FlatIndex, HNSWIndex, clustered_vectors, recall_at_k
from byoai.zoo import story_model

DIM = 32
queries = clustered_vectors(100, DIM, seed=2)

# ---- 1. How much work a search does, as the collection grows ----------------------------------------------
print("searching 100 queries for their 10 nearest neighbours (HNSW: M=12, ef=64)")
print(f"  {'vectors':>8} {'build':>7}   {'flat looks at':>13}   {'HNSW looks at':>13}   {'recall@10':>9}")
for n in (1000, 2000, 4000, 8000):
    data = clustered_vectors(n, DIM, seed=1)
    flat = FlatIndex(DIM)
    flat.add(data)
    hnsw = HNSWIndex(DIM, M=12, ef_construction=80, seed=0)
    t0 = time.perf_counter()
    hnsw.add(data)
    build = time.perf_counter() - t0
    exact = [flat.search(q, 10)[0] for q in queries]
    hnsw.distance_computations = 0
    found = [hnsw.search(q, 10, ef=64)[0] for q in queries]
    looked = hnsw.distance_computations / len(queries)
    print(f"  {n:>8} {build:>6.1f}s   {n:>13}   {looked:>8.0f} ({looked / n:4.0%})   {recall_at_k(found, exact):>9.3f}")

# ---- 2. The knob: ef ----------------------------------------------------------------------------------------
print(f"\nthe recall/speed knob, on the {n} vectors:")
print(f"  {'ef':>5} {'recall@10':>10} {'distances':>10} {'ms/query':>9}")
for ef in (10, 16, 32, 64, 128, 256):
    hnsw.distance_computations = 0
    t0 = time.perf_counter()
    found = [hnsw.search(q, 10, ef=ef)[0] for q in queries]
    ms = (time.perf_counter() - t0) / len(queries) * 1000
    print(f"  {ef:>5} {recall_at_k(found, exact):>10.3f} {hnsw.distance_computations / len(queries):>10.0f} {ms:>9.2f}")
t0 = time.perf_counter()
for q in queries:
    flat.search(q, 10)
print(f"  {'flat':>5} {1.0:>10.3f} {n:>10} {(time.perf_counter() - t0) / len(queries) * 1000:>9.2f}")
print("  (flat is one numpy product and wins on time at this size; HNSW's advantage is the work it skips,")
print("   which is what decides it at a million vectors — and in C++ each of its steps is ~100x cheaper)")

# ---- 3. The neighbour heuristic -------------------------------------------------------------------------------
subset = clustered_vectors(2000, DIM, seed=1)
small = FlatIndex(DIM)
small.add(subset)
exact_small = [small.search(q, 10)[0] for q in queries]
print("\nwhich links to keep (2000 vectors, ef=32):")
for name in ("the nearest M", "the paper's heuristic"):
    h = HNSWIndex(DIM, M=12, ef_construction=80, seed=0)
    if name == "the nearest M":
        h._select_neighbors = lambda q, cands, m: [c for _, c in cands[:m]]
    h.add(subset)
    print(f"  {name:<22} recall@10 {recall_at_k([h.search(q, 10, ef=32)[0] for q in queries], exact_small):.3f}")

# ---- 4. Searching text: what is in an embedding? -----------------------------------------------------------
model, tok = story_model(verbose=True)
engine = Engine(model)
sentences = sorted({s.strip() + "." for s in re.split(r"[.\n]", stories(400, seed=5).replace('"', "")) if len(s.strip()) > 15})
words = sorted({w for s in sentences for w in re.findall(r"[a-z]+", s.lower())})
column = {w: i for i, w in enumerate(words)}
idf = np.log(len(sentences) / np.array([sum(w in re.findall(r"[a-z]+", s.lower()) for s in sentences) for w in words]))


def model_vector(text: str) -> np.ndarray:
    return engine.embed(tok.encode(text))


def word_vector(text: str) -> np.ndarray:
    """Bag of words, weighted by rarity (TF-IDF): 'river' counts for more than 'the'."""
    v = np.zeros(len(words))
    for w, c in Counter(re.findall(r"[a-z]+", text.lower())).items():
        if w in column:
            v[column[w]] = c * idf[column[w]]
    return v


indexes = {}
for name, embed in (("model embedding", model_vector), ("bag of words", word_vector)):
    vectors = np.stack([embed(s) for s in sentences])
    index = FlatIndex(vectors.shape[1])
    index.add(vectors)
    indexes[name] = (index, embed)

print(f"\nsearching {len(sentences)} sentences two ways (top 3):")
for query in ("The fox found a key near the river.", "the owl and the key"):
    print(f"\n  query: {query!r}")
    for name, (index, embed) in indexes.items():
        ids, scores = index.search(embed(query), 3)
        print(f"    {name}:")
        for i, s in zip(ids, scores):
            print(f"      {s:.3f}  {sentences[i]}")
