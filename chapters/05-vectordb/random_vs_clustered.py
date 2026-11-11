"""Chapter 5: what random vectors do to HNSW, compared with embedding-like clustered ones.

    uv run python chapters/05-vectordb/random_vs_clustered.py        # ~20 seconds
"""

from __future__ import annotations

import numpy as np

from byoai.vectordb import FlatIndex, HNSWIndex, clustered_vectors, recall_at_k

DIM = 32
data_sets = {
    "clustered": lambda n, seed: clustered_vectors(n, DIM, seed=seed),
    "random": lambda n, seed: np.random.default_rng(seed).standard_normal((n, DIM)),
}
print("3,000 vectors, 100 queries, recall@10 and distance computations per query (M=12)")
for name, make in data_sets.items():
    data, queries = make(3000, 1), make(100, 2)
    flat = FlatIndex(DIM)
    flat.add(data)
    exact = [flat.search(q, 10)[0] for q in queries]
    index = HNSWIndex(DIM, M=12, ef_construction=80)
    index.add(data)
    row = []
    for ef in (16, 64):
        index.distance_computations = 0
        found = [index.search(q, 10, ef=ef)[0] for q in queries]
        row.append(f"ef={ef}: recall {recall_at_k(found, exact):.3f}, {index.distance_computations // len(queries):>4} distances")
    print(f"  {name:<10} " + "   ".join(row))
