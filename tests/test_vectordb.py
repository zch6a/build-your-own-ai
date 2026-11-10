import numpy as np
import pytest

from byoai.vectordb import FlatIndex, HNSWIndex, normalize, recall_at_k
from byoai.vectordb import clustered_vectors as clustered


def test_flat_index_is_exact():
    data = clustered(500, 16)
    index = FlatIndex(16)
    index.add(data)
    q = data[42] + 0.01
    ids, scores = index.search(q, k=5)
    assert ids[0] == 42
    brute = normalize(data) @ normalize(q)
    assert np.allclose(scores, np.sort(brute)[::-1][:5], atol=1e-6)
    assert list(scores) == sorted(scores, reverse=True)


@pytest.fixture(scope="module")
def built():
    data = clustered(3000, 32, seed=1)
    hnsw = HNSWIndex(32, M=12, ef_construction=80, seed=0)
    hnsw.add(data)
    flat = FlatIndex(32)
    flat.add(data)
    queries = clustered(100, 32, seed=2)
    return data, hnsw, flat, queries


def test_hnsw_recall_is_high(built):
    # recall@10 is strict here: the 10th and 11th true neighbours differ by ~0.003 in cosine.
    _, hnsw, flat, queries = built
    exact = [flat.search(q, 10)[0] for q in queries]
    assert recall_at_k([hnsw.search(q, 10, ef=64)[0] for q in queries], exact) > 0.9
    assert recall_at_k([hnsw.search(q, 10, ef=128)[0] for q in queries], exact) > 0.97


def test_the_neighbour_heuristic_beats_keeping_the_nearest(built):
    data, hnsw, flat, queries = built
    naive = HNSWIndex(32, M=12, ef_construction=80, seed=0)
    naive._select_neighbors = lambda q, cands, m: [c for _, c in cands[:m]]
    naive.add(data[:1500])
    smart = HNSWIndex(32, M=12, ef_construction=80, seed=0)
    smart.add(data[:1500])
    small = FlatIndex(32)
    small.add(data[:1500])
    exact = [small.search(q, 10)[0] for q in queries]
    r_naive = recall_at_k([naive.search(q, 10, ef=32)[0] for q in queries], exact)
    r_smart = recall_at_k([smart.search(q, 10, ef=32)[0] for q in queries], exact)
    assert r_smart > r_naive + 0.05, (r_smart, r_naive)


def test_more_ef_never_hurts_much_and_usually_helps(built):
    _, hnsw, flat, queries = built
    exact = [flat.search(q, 10)[0] for q in queries]
    low = recall_at_k([hnsw.search(q, 10, ef=10)[0] for q in queries], exact)
    high = recall_at_k([hnsw.search(q, 10, ef=200)[0] for q in queries], exact)
    assert high >= low and high > 0.99


def test_hnsw_looks_at_a_fraction_of_the_vectors(built):
    _, hnsw, _, queries = built
    hnsw.distance_computations = 0
    for q in queries:
        hnsw.search(q, 10, ef=64)
    per_query = hnsw.distance_computations / len(queries)
    assert per_query < 0.25 * len(hnsw), per_query


def test_every_stored_vector_finds_itself(built):
    data, hnsw, _, _ = built
    found = sum(hnsw.search(data[i], 1)[0][0] == i for i in range(0, 3000, 30))
    assert found >= 98


def test_layers_thin_out(built):
    _, hnsw, _, _ = built
    per_layer = [sum(len(l) > layer for l in hnsw.links) for layer in range(hnsw.top_layer + 1)]
    assert per_layer[0] == 3000
    assert all(a > b for a, b in zip(per_layer, per_layer[1:])), per_layer


def test_links_are_capped(built):
    _, hnsw, _, _ = built
    assert all(len(l[0]) <= hnsw.M0 for l in hnsw.links)
    assert all(len(nb) <= hnsw.M for l in hnsw.links for nb in l[1:])


def test_deleted_vectors_are_never_returned():
    data = clustered(400, 8, seed=3)
    hnsw = HNSWIndex(8, M=8, seed=1)
    hnsw.add(data)
    target = 17
    assert hnsw.search(data[target], 1)[0][0] == target
    hnsw.delete(target)
    ids, _ = hnsw.search(data[target], 10)
    assert target not in ids and len(ids) == 10 and len(hnsw) == 399


def test_save_and_load(tmp_path, built):
    _, hnsw, _, queries = built
    path = tmp_path / "index.pkl"
    hnsw.save(path)
    loaded = HNSWIndex.load(path)
    for q in queries[:10]:
        assert np.array_equal(loaded.search(q, 10)[0], hnsw.search(q, 10)[0])
    loaded.add(queries[:5])  # still usable for inserts
    assert len(loaded) == len(hnsw) + 5


def test_empty_and_tiny_indexes():
    hnsw = HNSWIndex(4)
    assert len(hnsw.search(np.ones(4), 3)[0]) == 0
    hnsw.add(np.eye(4)[:2])
    ids, scores = hnsw.search(np.array([1.0, 0, 0, 0]), 5)
    assert list(ids) == [0, 1] and np.isclose(scores[0], 1.0)
