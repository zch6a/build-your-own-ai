# Chapter 5 — A vector database: exact search, then HNSW

> **Code:** [`byoai/vectordb.py`](../../byoai/vectordb.py) (about 250 lines) · **Tests:** [`tests/test_vectordb.py`](../../tests/test_vectordb.py) · **Run:** `uv run python chapters/05-vectordb/run.py` (about 30 seconds)
>
> [← Chapter 4](../04-inference/README.md) · [All chapters](../../README.md#chapters) · [Chapter 6 →](../06-rag/README.md)

RAG, recommendations, deduplication and "more like this" all ask the same question: *which stored
vectors are closest to this one?* This chapter answers it twice: first exactly, then the way every
vector database does it at scale, with an approximate graph index (HNSW) that finds nearly the same
neighbours while looking at a small fraction of the vectors. Then it puts text in, and finds out what
an embedding actually captures.

## Similarity is a dot product

Vectors are normalized to length 1 when they are added. For unit vectors the dot product *is* the
cosine of the angle between them: 1 for the same direction, 0 for unrelated, −1 for opposite. Searching
then means "largest dot product", and for HNSW, which thinks in distances, distance is 1 − cosine.

## Exact search: one matrix product

```python
scores = self.vectors @ normalize(query)           # (n,)  — every stored vector at once
top = np.argpartition(-scores, k - 1)[:k]          # the k largest, unordered: O(n)
top = top[np.argsort(-scores[top])]                # order just those k
```

`FlatIndex` is the whole of it. It is exact, simple and, with numpy, fast: 8,000 vectors in 0.06 ms.
It is also the ground truth every approximate index is measured against. Its cost grows with n: a
billion vectors is a billion dot products per query.

## HNSW: a map with motorways

HNSW (Hierarchical Navigable Small World, Malkov & Yashunin 2016) builds a graph: every vector is a
node linked to some of its near neighbours. To search, start somewhere, look at the current node's
neighbours, move to whichever is closest to the query, repeat. Two ideas make this work:

* **Layers.** Every node is in layer 0; a random few are also in layer 1, fewer in layer 2, and so on
  (each layer about M times smaller). The top layers have few nodes and therefore long links — the
  motorways. A search starts at the top, gets close quickly, drops a layer, and finishes on the local
  roads of layer 0.
* **A beam, not a single walker.** On layer 0 the search keeps the `ef` best nodes found so far and
  expands the closest unexpanded one, until none of the candidates can improve the list. A greedy
  walker stops at the first node with no better neighbour (a local minimum); a beam of 64 rarely does.

```python
while candidates:
    dist_c, c = heappop(candidates)                    # closest unexpanded node
    if dist_c > worst_of(best) and len(best) >= ef:
        break                                          # nothing left can improve the list
    for nb in links[c][layer]:                         # (unvisited neighbours, one matrix product)
        if len(best) < ef or dist(nb) < worst_of(best):
            heappush(candidates, nb); heappush(best, nb)   # and drop the worst if over ef
```

Inserting is searching: find the new vector's nearest nodes on each of its layers, link it to M of
them in both directions, and if a neighbour now has too many links (more than M, or 2·M on layer 0),
prune it. (The first version gave new nodes 2·M links on layer 0 straight away; the paper gives M and
lets the backlinks fill nodes up to the 2·M cap. Following the paper built the index faster *and*
raised recall — the tables below are from the corrected version.)

How good is an approximate answer? **Recall@10**: of the true 10 nearest neighbours (found by
`FlatIndex`), the share that HNSW returned. 0.9 means it found 9 of the 10, on average over the queries.

## Which links to keep

The obvious rule — keep the M nearest — is not the one HNSW uses. In clustered data the M nearest are
all in the same cluster, often on the same side, and the graph falls apart into islands a search
cannot leave. The paper's heuristic takes candidates nearest first but **skips a candidate that is
closer to an already-chosen neighbour than to the new node**: that direction is already covered. What
is left are links that point different ways, including the rare one to the next cluster.

```text
which links to keep (2000 vectors, ef=32):
  the nearest M          recall@10 0.719
  the paper's heuristic  recall@10 0.927
```

Same number of links, same search: 21 points of recall from choosing them well. (The first version
computed each candidate pair's distance with its own numpy call; building took 38 s for 3,000 vectors.
Computing all pairwise distances in one product and looping over plain floats brought it to 10.)

## Random vectors are the wrong test data

The obvious test data — random Gaussian vectors — is nothing like what a model produces. Real
embeddings lie in clusters on a low-dimensional surface; random vectors in 32 dimensions fill the
space evenly, and every point is nearly the same distance from every other. `clustered_vectors` makes
the realistic kind (an 8-dimensional latent space, curved into 32 dimensions by a `tanh`), and the tests
use it. What does random data do to HNSW? Measured on 3,000 vectors, 100 queries:

```text
               ef=16                        ef=64
clustered      recall 0.838,  171 distances  recall 0.935,  336 distances
random         recall 0.780,  351 distances  recall 0.980,  879 distances
```

Not what this chapter's first draft claimed ("noise is the hardest case, recall collapses"). Recall
holds up — but the search does two to three times the work for it: with no structure to follow, the
beam wanders before it settles. At the same work (about 350 distances), clustered data gets 0.94 and
random data 0.78. An index tuned on random vectors would look slow and over-provisioned; one tested
only on random vectors would never meet the clustered, uneven data where the neighbour heuristic
above earns its keep.

Recall@10 is a strict measure on both: the 10th and 11th true neighbours typically differ by 0.002–0.004
in cosine, and finding the 11th instead counts as a miss.
## What the demo shows

How much work a search does as the collection grows:

```text
 vectors   flat looks at   HNSW looks at   recall@10
    1000            1000        300 ( 30%)       0.991
    2000            2000        329 ( 16%)       0.984
    4000            4000        380 ( 10%)       0.918
    8000            8000        414 (  5%)       0.947
```

Eight times the vectors, 1.4 times the work: HNSW's cost grows roughly with log n, flat search's with
n. (Recall at a fixed `ef` moves a few points from graph to graph; each build is a different random
graph.) Then the knob:

```text
   ef  recall@10  distances  ms/query
   10      0.766        164      0.17
   32      0.918        296      0.33
   64      0.947        414      0.53
  128      0.977        622      0.88
  256      0.993        943      1.51
 flat      1.000       8000      0.06
```

Be honest about the last line: at 8,000 vectors, numpy's single matrix product beats a Python graph
walk on time. HNSW's advantage is the work it *skips*, and that is what decides it at a million
vectors, where flat search does a million dot products and HNSW, going by the table, a few thousand
at most. Libraries
like hnswlib and FAISS run the same algorithm in C++, where each step costs nanoseconds, not
microseconds.

## Searching text: what is in an embedding?

The last part embeds 1,196 sentences of the story corpus with the chapter-3 model — the average of its
final hidden states — and searches them. For comparison, it also searches with the oldest trick there
is: a bag of words weighted by rarity (TF-IDF).

```text
query: 'the owl and the key'
  model embedding:
    0.859  It was cold and the cat was tired.
    0.859  It was cold and the bear was tired.
    0.859  It was warm and the bear was tired.
  bag of words:
    0.723  The owl found a key near the market.
    0.492  Tom gave Noor a key.
    0.488  Mia gave Tom a key.
```

The model finds sentences of the same *shape* — "… and the [animal] …" — and ignores the owl and the
key. That is not a bug in the search; it is what the vector holds. The model was trained to predict the
next token, and for that the most useful thing to know is where in the grammar you are. Which animal
it is matters for exactly one prediction later on. So the hidden states mostly encode structure, and
every sentence of one template comes out at cosine ≈ 1.000 with every other (centring the vectors, a
common fix, does not change the order).

Real embedding models (the ones RAG systems use) start from a language model like this and are then
**trained for similarity**: pairs that should match — a question and its answer, a sentence and its
paraphrase — are pulled together, and other pairs in the batch pushed apart (a contrastive loss). That
second training is what makes "the owl and the key" land next to sentences about owls and keys. The
lesson for the next chapter: retrieval quality is decided by the embedding, not the index. The
chapter-6 RAG system therefore starts with keyword search (BM25, a refined bag of words), which for
exact facts is hard to beat.

## Exercises

1. Set `M=4` and `M=32`. How do recall, build time and memory (count the links) change?
2. Delete half the vectors (`delete` marks them; they still route searches). What happens to recall
   and to the work per query? When would you rebuild?
3. Replace `clustered_vectors(n, DIM, seed=s)` in the demo with
   `np.random.default_rng(s).standard_normal((n, DIM))`. Compare recall *and* distance computations per
   query at the same `ef`. Where does the extra work go?
4. Make the model embedding better without training: use only the hidden state at the animal's
   position, or average only over nouns. How close to the bag of words can you get?
5. Add product quantization: split each vector into 8 pieces, replace each piece by the nearest of
   256 learned centroids, and store one byte per piece. How much recall do you lose for 16× less
   memory?

## Where real systems differ

Production vector databases add what this chapter leaves out: memory-mapped storage instead of an
in-memory pickle, compressed vectors (product or scalar quantization) so a billion vectors fit in RAM,
filters ("only documents from 2024") woven into the graph search, concurrent inserts, and real deletes
that repair the graph instead of marking nodes. Some use other indexes — IVF (cluster first, search a
few clusters), or DiskANN for graphs on SSD — but HNSW is the default in most of them.

**Next:** [Chapter 6 — RAG](../06-rag/README.md): retrieve the right passages, put them in the prompt, and make a model answer
from them instead of from memory.
