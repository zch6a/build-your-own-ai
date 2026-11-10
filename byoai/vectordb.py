"""Chapter 5 — a vector database: exact search, then HNSW.

Retrieval for RAG, recommendations, deduplication: they all ask "which stored vectors are closest to
this one?". With a few thousand vectors, compare against all of them. With millions, that is too slow,
and every vector database answers it the same way — an *approximate* index that finds almost the same
neighbours while looking at a tiny fraction of the vectors. HNSW (Hierarchical Navigable Small World)
is the one most of them use.

    >>> from byoai.vectordb import FlatIndex, HNSWIndex
    >>> index = HNSWIndex(dim=64)
    >>> index.add(vectors)                       # (n, 64)
    >>> ids, scores = index.search(query, k=10)

Similarity is cosine: vectors are normalized when added, so a dot product is the cosine.
"""

from __future__ import annotations

import heapq
import math
import pickle
from pathlib import Path

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """Unit length along the last axis (a zero vector stays zero)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1, n)


class FlatIndex:
    """Exact search: one matrix product against every stored vector. The ground truth for HNSW."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray) -> list[int]:
        v = normalize(np.atleast_2d(vectors))
        start = len(self.vectors)
        self.vectors = np.vstack([self.vectors, v])
        return list(range(start, start + len(v)))

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """The k most similar stored vectors: (ids, cosine similarities), best first."""
        scores = self.vectors @ normalize(query)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return top, scores[top]


class HNSWIndex:
    """Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016).

    Every vector is a node in layer 0; a few are also in layer 1, fewer in layer 2, and so on — each
    layer a graph linking nodes to near neighbours. A search starts at the top, where few long links
    cross the space quickly, walks greedily towards the query, drops a layer, and repeats; at the
    bottom it explores more widely (`ef_search` candidates) and returns the best k.

    * `M`: links per node per layer (layer 0 gets 2·M). More links: better recall, more memory.
    * `ef_construction`: how widely to search for neighbours while inserting. Better graph, slower build.
    * `ef_search`: how widely to search at query time. The recall/speed knob.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100, ef_search: int = 50, seed: int = 0):
        self.dim, self.M, self.M0 = dim, M, 2 * M
        self.ef_construction, self.ef_search = ef_construction, ef_search
        self.level_mult = 1 / math.log(M)  # levels drop by a factor of M per layer
        self.rng = np.random.default_rng(seed)
        self._vec = np.zeros((1024, dim), dtype=np.float32)  # grown by doubling
        self.n = 0
        self.links: list[list[list[int]]] = []  # links[node][layer] -> neighbour ids
        self.deleted: set[int] = set()
        self.entry: int | None = None
        self.top_layer = -1
        self.distance_computations = 0  # for the demo: how much work a search did

    def __len__(self) -> int:
        return self.n - len(self.deleted)

    @property
    def vectors(self) -> np.ndarray:
        return self._vec[: self.n]

    # ---- distances: 1 − cosine on unit vectors ---------------------------------------------------------
    def _dist(self, q: np.ndarray, ids: list[int] | np.ndarray) -> np.ndarray:
        self.distance_computations += len(ids)
        return 1.0 - self._vec[ids] @ q

    # ---- searching one layer ---------------------------------------------------------------------------
    def _search_layer(self, q: np.ndarray, entries: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Best-first search of one layer from `entries`; returns up to `ef` (distance, id), nearest first.

        Two heaps: `candidates` (nearest first) to expand next, `best` (farthest first, size ef) found so
        far. Stop when the nearest unexpanded candidate is farther than the worst of the best.
        """
        visited = set(entries)
        d = self._dist(q, entries)
        candidates = [(float(di), e) for di, e in zip(d, entries)]
        heapq.heapify(candidates)
        best = [(-float(di), e) for di, e in zip(d, entries)]
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)
        while candidates:
            dist_c, c = heapq.heappop(candidates)
            if dist_c > -best[0][0] and len(best) >= ef:
                break
            fresh = [nb for nb in self.links[c][layer] if nb not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for dn, nb in zip(self._dist(q, fresh), fresh):  # all neighbours in one matrix product
                dn = float(dn)
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(candidates, (dn, nb))
                    heapq.heappush(best, (-dn, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-nd, i) for nd, i in best)

    def _select_neighbors(self, q: np.ndarray, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """The paper's heuristic: take candidates nearest first, but skip one that is closer to an
        already chosen neighbour than to `q` — it would add a link in a direction already covered.
        Keeps links spread out, which keeps the graph navigable across clusters."""
        ids = [c for _, c in candidates]
        dists = [d for d, _ in candidates]
        # All pairwise distances between candidates in one product (then plain Python: a million tiny
        # numpy calls, one per pair, made building 15x slower).
        vecs = self._vec[ids]
        pair = (1.0 - vecs @ vecs.T).tolist()
        chosen: list[int] = []  # positions in `ids`
        for j in range(len(ids)):
            if len(chosen) >= m:
                break
            row = pair[j]
            if any(row[i] < dists[j] for i in chosen):
                continue
            chosen.append(j)
        # Fill up with the nearest skipped ones, so a node never ends up with very few links.
        if len(chosen) < m:
            taken = set(chosen)
            chosen += [j for j in range(len(ids)) if j not in taken][: m - len(chosen)]
        return [ids[j] for j in chosen]

    # ---- inserting -------------------------------------------------------------------------------------
    def add(self, vectors: np.ndarray) -> list[int]:
        """Insert vectors; returns their ids (0, 1, 2, … in insertion order)."""
        return [self._insert(v) for v in normalize(np.atleast_2d(vectors))]

    def _insert(self, v: np.ndarray) -> int:
        if self.n == len(self._vec):
            self._vec = np.vstack([self._vec, np.zeros_like(self._vec)])
        node = self.n
        self._vec[node] = v
        self.n += 1
        level = int(-math.log(1.0 - self.rng.random()) * self.level_mult)
        self.links.append([[] for _ in range(level + 1)])
        if self.entry is None:
            self.entry, self.top_layer = node, level
            return node
        # Greedy descent through the layers above this node's top.
        entries = [self.entry]
        for layer in range(self.top_layer, level, -1):
            entries = [self._search_layer(v, entries, 1, layer)[0][1]]
        # From its top layer down: find neighbours, link both ways, prune.
        for layer in range(min(level, self.top_layer), -1, -1):
            found = self._search_layer(v, entries, self.ef_construction, layer)
            m = self.M0 if layer == 0 else self.M
            neighbours = self._select_neighbors(v, found, m)
            self.links[node][layer] = neighbours
            for nb in neighbours:
                nb_links = self.links[nb][layer]
                nb_links.append(node)
                if len(nb_links) > m:
                    # Too many links: keep the best m by the same heuristic.
                    d = self._dist(self._vec[nb], nb_links)
                    self.links[nb][layer] = self._select_neighbors(self._vec[nb], sorted(zip(d.tolist(), nb_links)), m)
            entries = [i for _, i in found]
        if level > self.top_layer:
            self.entry, self.top_layer = node, level
        return node

    def delete(self, node: int) -> None:
        """Mark a node deleted: it still routes searches (the graph stays connected) but is never returned."""
        if not 0 <= node < self.n:
            raise KeyError(node)
        self.deleted.add(node)

    # ---- searching -------------------------------------------------------------------------------------
    def search(self, query: np.ndarray, k: int = 10, ef: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Approximately the k most similar vectors: (ids, cosine similarities), best first."""
        if self.entry is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        q = normalize(query)
        ef = max(ef or self.ef_search, k)
        entries = [self.entry]
        for layer in range(self.top_layer, 0, -1):
            entries = [self._search_layer(q, entries, 1, layer)[0][1]]
        # Deleted nodes take up room in the result: search a little wider so k live ones remain.
        found = self._search_layer(q, entries, ef + len(self.deleted & set(range(self.n))) if self.deleted else ef, 0)
        live = [(d, i) for d, i in found if i not in self.deleted][:k]
        return np.array([i for _, i in live], dtype=np.int64), np.array([1.0 - d for d, _ in live], dtype=np.float32)

    # ---- saving ----------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        state = {k: v for k, v in self.__dict__.items() if k not in ("rng", "_vec")}
        state["vectors"] = self.vectors.copy()
        state["rng_state"] = self.rng.bit_generator.state
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> HNSWIndex:
        """Load an index saved by `save` (pickle: only load files you made yourself)."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        index = cls(state["dim"], state["M"], state["ef_construction"], state["ef_search"])
        vectors = state.pop("vectors")
        index.rng.bit_generator.state = state.pop("rng_state")
        index.__dict__.update(state)
        index._vec = np.zeros((max(1024, len(vectors)), index.dim), dtype=np.float32)
        index._vec[: len(vectors)] = vectors
        return index


def clustered_vectors(n: int, dim: int, clusters: int = 20, latent: int = 8, seed: int = 0) -> np.ndarray:
    """Synthetic vectors shaped like real embeddings: in clusters, on a curved low-dimensional surface
    (`latent` dimensions mapped into `dim`). Pure random noise in many dimensions is the hardest case
    for any approximate index — and nothing like what a model produces."""
    r = np.random.default_rng(seed)
    centers = r.standard_normal((clusters, latent))
    points = centers[r.integers(0, clusters, n)] + 0.35 * r.standard_normal((n, latent))
    return np.tanh(points @ r.standard_normal((latent, dim)))


def recall_at_k(approx_ids: list[np.ndarray], exact_ids: list[np.ndarray]) -> float:
    """The share of the true k nearest neighbours that the approximate search found."""
    hits = sum(len(set(a.tolist()) & set(e.tolist())) for a, e in zip(approx_ids, exact_ids))
    return hits / sum(len(e) for e in exact_ids)
